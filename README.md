# ChlCast

ChlCast forecasts surface chlorophyll-a for the next 10 days on a 4 km grid from 60°S to 60°N. It uses 10 days of observations and daily climatology.

## Quick start

Download both files from the v1.0.0 release.

- [Linux x86_64 environment](https://github.com/hohosilent-hoho/ChlCast/releases/download/v1.0.0/ChlCast_env_linux_x86_64.tar.gz)
- [August 25, 2026 example inputs](https://github.com/hohosilent-hoho/ChlCast/releases/download/v1.0.0/ChlCast_example_2026-08-25.zip)

On Linux x86_64, save both files in the project directory and run the following commands.

```bash
mkdir -p chlcast_env example_files
tar -xzf ChlCast_env_linux_x86_64.tar.gz -C chlcast_env
unzip ChlCast_example_2026-08-25.zip -d example_files
./chlcast_env/bin/python ./chlcast_env/bin/conda-unpack

./chlcast_env/bin/chlcast-python inference.py \
  --input-dir example_files/2026-08-25 \
  --device cuda --batch-size 4 \
  --output ChlCast_2026-08-25.nc
```

Use `--device cpu` and omit `--batch-size 4` for CPU inference. The example writes one NetCDF file with daily chlorophyll-a forecasts and a validity mask for August 26 through September 4, 2026.

## Input data

Each initialization folder contains two NetCDF files with `CHL(time, latitude, longitude)` in `mg m-3` on the same grid.

- `observations/chlorophyll_observations.nc` covers the initialization date and the previous nine days.
- `climatology/chlorophyll_climatology.nc` covers those ten days and all ten forecast dates.

The observations come from the daily gap-free 4 km Copernicus Marine Global Ocean Colour L4 product (`OCEANCOLOUR_GLO_BGC_L4_MY_009_104`). [Product and data access](https://data.marine.copernicus.eu/product/OCEANCOLOUR_GLO_BGC_L4_MY_009_104/services) · [DOI 10.48670/moi-00281](https://doi.org/10.48670/moi-00281)

For each calendar day and grid cell, compute the arithmetic mean of valid chlorophyll concentrations within a circular 31-day window (±15 days) across prior years. The 2026 example uses 1998–2025.
