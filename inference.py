#!/usr/bin/env python3
"""Run a ten-day 2+1D U-Net M2 chlorophyll forecast with ONNX Runtime.

Install numpy, xarray, h5netcdf, and onnxruntime (onnxruntime-gpu for CUDA).

Example:
    python inference.py --input-dir example_files/2026-08-31 \
        --model ChlCast.onnx --output forecast_2026-08-31.nc

The model accepts float32 history tiles [B, 2, 10, 288, 288], history
time features [B, 10, 4], and target time features [B, 5, 4]. It
returns five log-chlorophyll anomaly fields per tile [B, 5, 256, 256].
"""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5netcdf
import numpy as np
import xarray as xr


HISTORY_DAYS = 10
STAGE_DAYS = 5
FORECAST_DAYS = 2 * STAGE_DAYS
CENTER = 256
HALO = 16
TILE_SIZE = CENTER + 2 * HALO
HEIGHT, WIDTH = 2880, 8640
EPSILON = 0.01
UNITS = "milligram m-3"
GRID_AXES = ("latitude", "longitude")
DAY = np.timedelta64(1, "D")


def input_times(dataset, path):
    if dataset.CHL.dims != ("time", *GRID_AXES):
        raise ValueError(f"{path}: CHL must have dimensions (time, latitude, longitude)")
    if dataset.CHL.attrs.get("units") != UNITS:
        raise ValueError(f"{path}: CHL units must be {UNITS}")
    for name, default in (("scale_factor", 1), ("add_offset", 0)):
        if float(np.asarray(dataset.CHL.attrs.get(name, default)).item()) != default:
            raise ValueError(f"{path}: packed CHL is unsupported")
    if dataset.time.dims != ("time",):
        raise ValueError(f"{path}: time must be a one-dimensional coordinate")
    days = np.asarray(dataset.time.values, dtype="datetime64[D]")
    if not len(days) or np.isnat(days).any() or np.any(days[1:] <= days[:-1]):
        raise ValueError(f"{path}: time must contain unique dates in increasing order")
    return days


def grid_coordinates(observations, climatology):
    coordinates = []
    for axis, size, lower_edge in zip(GRID_AXES, (HEIGHT, WIDTH), (-60, -180)):
        observed = np.asarray(observations[axis].values, dtype=np.float32)
        climate = np.asarray(climatology[axis].values, dtype=np.float32)
        expected = lower_edge + (np.arange(size) + 0.5) / 24
        if (observations[axis].dims != (axis,) or climatology[axis].dims != (axis,)
                or observed.shape != (size,) or not np.array_equal(observed, climate)
                or not np.allclose(observed, expected, rtol=0, atol=3e-5)):
            raise ValueError(f"Input {axis} coordinates must match on the native 4 km grid (60°S–60°N)")
        coordinates.append(observed)
    return coordinates


def select_days(dataset, available, required, label):
    first = int(np.searchsorted(available, required[0]))
    last = first + len(required)
    if not np.array_equal(available[first:last], required):
        missing = np.setdiff1d(required, available)
        raise ValueError(f"{label} is missing required dates: {', '.join(map(str, missing))}")
    return dataset.CHL.isel(time=slice(first, last))


def encode(field, valid):
    encoded = np.full(field.shape, np.nan, dtype=np.float32)
    encoded[valid] = np.log(field[valid] + EPSILON)
    return encoded


def read_log_field(field, index):
    values = np.asarray(field.isel(time=index).values, dtype=np.float32)
    valid = np.isfinite(values) & (values >= 0)
    for attr in ("_FillValue", "missing_value"):
        for fill in np.asarray(field.attrs.get(attr, []), dtype=np.float32).flat:
            if fill == 0:
                raise ValueError("A CHL fill value of zero would discard valid ocean zeros")
            valid &= values != fill
    return encode(values, valid), valid


def as_anomaly(encoded, valid, climate):
    climate_values, climate_valid = climate
    encoded -= climate_values
    encoded[~valid | ~climate_valid] = 0
    np.nan_to_num(encoded, copy=False, nan=0, posinf=0, neginf=0)
    return encoded, valid


def load_fields(observations, climatology):
    climates = []
    for index in range(HISTORY_DAYS + FORECAST_DAYS):
        encoded, valid = read_log_field(climatology, index)
        np.nan_to_num(encoded, copy=False, nan=float(np.log(EPSILON)),
                      posinf=float(np.log(EPSILON)), neginf=float(np.log(EPSILON)))
        climates.append((encoded, valid))
    anomalies = [
        as_anomaly(*read_log_field(observations, index), climates[index])
        for index in range(HISTORY_DAYS)
    ]
    return anomalies, climates


def time_encodings(dates):
    years = dates.astype("datetime64[Y]")
    starts = years.astype("datetime64[D]")
    year_lengths = ((years + np.timedelta64(1, "Y")).astype("datetime64[D]") - starts)
    annual = 2.0 * np.pi * (dates - starts).astype(np.float32) / year_lengths.astype(np.float32)
    relative = 2.0 * np.pi * np.arange(len(dates), dtype=np.float32) / len(dates)
    encodings = np.stack((np.sin(annual), np.cos(annual), np.sin(relative), np.cos(relative)), axis=-1)
    return encodings[:HISTORY_DAYS], encodings[HISTORY_DAYS:]


def load_model(path, device, requested_batch):
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {path}")
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install onnxruntime for CPU or onnxruntime-gpu for CUDA") from exc
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDA is unavailable in this ONNX Runtime installation")
        providers.insert(0, ("CUDAExecutionProvider", {"use_tf32": "0"}))
    session = ort.InferenceSession(str(path), providers=providers)
    if device == "cuda" and "CUDAExecutionProvider" not in session.get_providers():
        raise RuntimeError("ONNX Runtime could not initialize CUDA")

    inputs = session.get_inputs()
    outputs = session.get_outputs()
    shapes = ((2, HISTORY_DAYS, TILE_SIZE, TILE_SIZE), (HISTORY_DAYS, 4), (STAGE_DAYS, 4))
    if len(inputs) != len(shapes) or len(outputs) != 1:
        raise ValueError("Expected the M2 ONNX model with three inputs and one output")
    names = []
    for shape in shapes:
        matches = [item for item in inputs
                   if item.type == "tensor(float)" and tuple(item.shape[1:]) == shape]
        if len(matches) != 1:
            raise ValueError(f"Expected a float32 ONNX input with shape [batch, {shape}]")
        names.append(matches[0].name)
    fixed_sizes = {item.shape[0] for item in [*inputs, outputs[0]]
                   if item.shape and isinstance(item.shape[0], int)}
    if len(fixed_sizes) > 1 or (fixed_sizes and min(fixed_sizes) < 1):
        raise ValueError("The ONNX model has inconsistent batch dimensions")
    fixed_batch = next(iter(fixed_sizes), None)
    if fixed_batch is not None and requested_batch is not None and fixed_batch != requested_batch:
        raise ValueError(f"This ONNX model requires --batch-size {fixed_batch}")
    return session, names, outputs[0].name, fixed_batch or requested_batch or 1, fixed_batch


def predict_stage(model, anomalies, climates, dates):
    session, names, output_name, batch_size, fixed_batch = model
    history_time, future_time = time_encodings(dates)
    height, width = anomalies[0][0].shape
    tiles = [(top, left) for top in range(0, height, CENTER) for left in range(0, width, CENTER)]
    forecast = np.empty((STAGE_DAYS, height, width), dtype=np.float32)
    buffer = np.empty((batch_size, 2, HISTORY_DAYS, TILE_SIZE, TILE_SIZE), dtype=np.float32)
    offsets = np.arange(-HALO, CENTER + HALO)

    for start in range(0, len(tiles), batch_size):
        batch = tiles[start:start + batch_size]
        for position, (top, left) in enumerate(batch):
            selection = np.ix_(np.clip(top + offsets, 0, height - 1), (left + offsets) % width)
            for day in range(HISTORY_DAYS):
                buffer[position, 0, day] = anomalies[day][0][selection]
                buffer[position, 1, day] = climates[day][0][selection]
        size = fixed_batch or len(batch)
        if size > len(batch):
            buffer[len(batch):size] = buffer[len(batch) - 1]
        inputs = (buffer[:size], np.repeat(history_time[None], size, axis=0),
                  np.repeat(future_time[None], size, axis=0))
        prediction, = session.run([output_name], dict(zip(names, inputs)))
        if (prediction.shape != (size, STAGE_DAYS, CENTER, CENTER)
                or prediction.dtype != np.float32 or not np.isfinite(prediction).all()):
            raise RuntimeError("Unexpected ONNX forecast shape, type, or non-finite values")
        for patch, (top, left) in zip(prediction, batch):
            rows, columns = min(CENTER, height - top), min(CENTER, width - left)
            forecast[:, top:top + rows, left:left + columns] = patch[:, :rows, :columns]

    history_valid = np.logical_and.reduce([frame[1] for frame in anomalies])
    history_valid &= climates[HISTORY_DAYS - 1][1]
    masks = np.empty(forecast.shape, dtype=bool)
    for lead, (climate, valid) in enumerate(climates[HISTORY_DAYS:]):
        masks[lead] = history_valid & valid
        values = forecast[lead]
        values += climate
        with np.errstate(over="ignore"):
            np.exp(values, out=values)
        values -= EPSILON
        np.maximum(values, 0, out=values)
        if not np.isfinite(values[masks[lead]]).all():
            raise RuntimeError("Non-finite CHL concentration on valid pixels")
        values[~masks[lead]] = np.nan
    return forecast, masks


@contextmanager
def forecast_file(path, origin, latitude, longitude, model_path):
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix=f".{path.name}.", suffix=".partial", dir=path.parent,
                            delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with h5netcdf.File(temporary, "w") as output:
            dims = ("forecast_reference_time", "lead_time", *GRID_AXES)
            output.dimensions = dict(zip(dims, (1, FORECAST_DAYS, len(latitude), len(longitude))))
            for axis, values, units in zip(GRID_AXES, (latitude, longitude),
                                           ("degrees_north", "degrees_east")):
                coord = output.create_variable(axis, (axis,), dtype="f4")
                coord[:] = values
                coord.attrs.update({"standard_name": axis, "units": units})
            epoch = np.datetime64("1970-01-01", "D")
            origin_days = int((origin - epoch) / DAY)
            leads = np.arange(1, FORECAST_DAYS + 1)
            calendar = {"units": "days since 1970-01-01", "calendar": "proleptic_gregorian"}
            reference = output.create_variable(dims[0], (dims[0],), dtype="i4")
            reference[:] = [origin_days]
            reference.attrs.update({**calendar, "standard_name": dims[0]})
            lead = output.create_variable(dims[1], (dims[1],), dtype="i2")
            lead[:] = leads
            lead.attrs.update({"units": "days", "standard_name": "forecast_period"})
            target = output.create_variable("target_time", dims[:2], dtype="i4")
            target[0] = origin_days + leads
            target.attrs.update({**calendar, "standard_name": "time"})
            options = dict(chunks=(1, 1, min(256, len(latitude)), min(512, len(longitude))),
                           compression="gzip", compression_opts=1, shuffle=True)
            chl = output.create_variable("CHL", dims, dtype="f4", fillvalue=np.float32(np.nan), **options)
            chl.attrs.update({"units": UNITS, "coordinates": "target_time",
                              "standard_name": "mass_concentration_of_chlorophyll_a_in_sea_water",
                              "ancillary_variables": "valid_ocean_mask"})
            mask = output.create_variable("valid_ocean_mask", dims, dtype="u1", **options)
            mask.attrs.update({"flag_values": np.array([0, 1], dtype=np.uint8),
                               "flag_meanings": "invalid valid"})
            output.attrs.update({"Conventions": "CF-1.8", "history_days": HISTORY_DAYS,
                                 "forecast_days": FORECAST_DAYS, "model_file": model_path.name,
                                 "history_start": str(origin - (HISTORY_DAYS - 1) * DAY),
                                 "history_end": str(origin),
                                 "rollout": "Stage two uses five observed days and five stage-one forecast days."})
            yield chl, mask
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args):
    observation_path = args.input_dir / "observations/chlorophyll_observations.nc"
    climate_path = args.input_dir / "climatology/chlorophyll_climatology.nc"
    options = dict(engine="h5netcdf", mask_and_scale=False, cache=False)
    with xr.open_dataset(observation_path, **options) as observed, xr.open_dataset(climate_path, **options) as climate:
        obs_days = input_times(observed, observation_path)
        climate_days = input_times(climate, climate_path)
        latitude, longitude = grid_coordinates(observed, climate)
        origin = np.datetime64(args.date, "D") if args.date else obs_days[-1]
        dates = origin + np.arange(1 - HISTORY_DAYS, FORECAST_DAYS + 1) * DAY
        observations = select_days(observed, obs_days, dates[:HISTORY_DAYS], "Observations")
        climatology = select_days(climate, climate_days, dates, "Climatology")
        if args.check_inputs:
            print(f"{origin}: observations {dates[0]} to {origin}, climatology {dates[0]} to {dates[-1]}; grid {HEIGHT} × {WIDTH}")
            return
        output = args.output or Path(f"2p1d_m2_10day_{origin}.nc")
        if output.exists():
            raise FileExistsError(output)
        model = load_model(args.model, args.device, args.batch_size)
        anomalies, climates = load_fields(observations, climatology)

    window = HISTORY_DAYS + STAGE_DAYS
    with forecast_file(output, origin, latitude, longitude, args.model) as (chl, valid_mask):
        first, first_mask = predict_stage(model, anomalies, climates[:window], dates[:window])
        chl[0, :STAGE_DAYS] = first
        valid_mask[0, :STAGE_DAYS] = first_mask
        continuation = anomalies[STAGE_DAYS:]
        del anomalies
        for lead, valid in enumerate(first_mask):
            continuation.append(as_anomaly(encode(first[lead], valid), valid, climates[HISTORY_DAYS + lead]))
        del first, first_mask
        print(f"{origin}: completed forecast days 1–5", flush=True)
        second, second_mask = predict_stage(model, continuation, climates[STAGE_DAYS:], dates[STAGE_DAYS:])
        chl[0, STAGE_DAYS:] = second
        valid_mask[0, STAGE_DAYS:] = second_mask
    print(f"{origin}: wrote forecast days 1–10 to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True, help="Folder containing observations/ and climatology/")
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().with_name("ChlCast.onnx"))
    parser.add_argument("--date", type=date.fromisoformat, help="Forecast origin, defaulting to the last observation date")
    parser.add_argument("--output", type=Path, help="Forecast NetCDF path")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, help="Tiles per model call, defaulting to one for a dynamic model")
    parser.add_argument("--check-inputs", action="store_true", help="Check dates and grid without loading the model")
    args = parser.parse_args()
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    try:
        run(args)
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
