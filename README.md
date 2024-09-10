# NoiseXMap

## Environment Setup
```
source /home/elise/elise_envs/miniconda3/bin/activate

conda create -n NoiseXMap python=3.10

conda activate NoiseXMap

conda install numpy pandas matplotlib scipy scikit-learn
conda install imageio

pip install cdsapi # required for ERA5 data
```

## ERA5 Reanalysis
Source website: [https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview](https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview).

ERA5 data extraction and processing scripts can be found under the folder "ERA5_extraction". Might need to modify path before usage.