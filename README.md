# Aug20_epri

For Team Epri at S2DS August 2020

## Data 

For the time being the data used in this project are synthetic data (generated by appropriate models) provided in `csv` format by EPRI. 

### How to Get and Extract the Data

For easier handling we use prepared zip-files which can be read without extracting them using the function `import_data.import_df_from_zip`. The prepared zip-files are located in our [google-drive](https://drive.google.com/drive/folders/1IByP1vFGRjsDTWvWETm523yIduE5Ao4E) and in EPRI's [box.com storage](https://app.box.com/folder/120323763205). It is also possible to use the extracted zip-folders (might be faster) using the function `import_data.import_df_from_dir`.

In order to prepare (rename and re-zip) these zip-files, bash scripts are available under `/tools/`. Usually, this should not be necessary as prepared zip-files are available. Use these scripts with caution as some of them rename all files in the destination directory.

### Types of Synthetic Data

(taken from Daniel's comments on the data)

- Basic - only deterministic losses (basic PV behavior) and degradation (labeled)
- Soil - soiling loss (labeled) in addition to degradation
- Weather - performance under real weather conditions (weather years shuffled randomly per dataset), degradation loss included but no soiling loss
- Soil Weather  - soiling loss added to Weather
- Soil Weather Locations - weather conditions are taken from 7 different locations (1 per dataset, included in filename) - weather years shuffled

### Dataset Specifications

There are no missing data points in these synthetic datasets. The datasets are given in the form of time series over 5 years in 1 minute intervals (2629440 data points). The timestamps of the data points are given in local time in the format `YYYY-MM-DD hh:mm:ssTimezone`, e.g., `2019-12-31 23:59:00-05:00 for UTC-05:00`.

For each data point of the time series the following attributes are provided:

| Attribute               | Column name             |  Units |
|-------------------------|:-----------------------:|-------:|
| Power                   | Power                   | kW     |
| Plane of array          | POA                     | W/m^2  |
| Ambient Temperatur      | Tamb                    | °C     |
| Wind                    | Wind                    | m/s    |
| Degradation rate / year | Degradation_rate_per_yr | %/year |
| Soiling                 | soiling                 | -      |
