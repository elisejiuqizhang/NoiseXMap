# xarray visualization (mars 2024 - Raphael)
path_to_your_file='/home/automation/elisejzh/Desktop/elisejzh/Projects/Mine/multivarCM/data_files/ERA5_data/longer_raw_nc/'
# Importing
import xarray as xr
import matplotlib.pyplot as plt
# Opening files and averaging for simplicity
ds = xr.open_dataset(path_to_your_file + 'slev_2013.nc')
ds = ds.mean(dim=('latitude','longitude')).compute()
# Plotting
ds['msnlwrf'].isel(time=slice(0,500)).plot.line('b')
ds['msnlwrfcs'].isel(time=slice(0,500)).plot.line(':b')
ds['msnswrf'].isel(time=slice(0,500)).plot.line('r')
ds['msnswrfcs'].isel(time=slice(0,500)).plot.line(':r')
plt.legend(['LW','LW, clear sky','Solar','Solar, clear sky'])
plt.ylabel('Radiation [W/m**2]')
plt.savefig('radiation.png')
