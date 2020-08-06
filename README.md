# Aug20_epri
For Team Epri at S2DS August 2020



## Data 

For the time being the data used in this project are synthetic data (generated by appropriate models) provided in `csv` format by EPRI. 

### How to Get and Extract the Data

Download the folder "Data" as a zip archive from this project's [box.com storage](https://app.box.com/folder/117614749421) provided by EPRI. Use the bash script at `tools/extract_and_rename_raw_synthetic_data.sh` to extract the data. 

Usage: `extract_and_rename_raw_synthetic_data.sh /path/to/datfile.zip /path/to/root/of/git-repo`, e.g. `extract_and_rename_raw_synthetic_data.sh ~/downloads/Data.zip ~/s2ds/code/Aug20_Epri/`.

MAKE SURE TO PROVIDE THE CORRECT PATH FOR THE SECOND ARGUMENT AS THE SCRIPT MIGHT RENAME FILES IN WHATEVER FOLDER YOU PROVIDE!

The script takes about 10 minutes to extract and rename all datasets. Folders and files are renamed according to the file structure shown below.

```
data
│
└───processed
│   
└───raw
    └───synthetic_basic
    │   │ synthetic_basic_001.csv
    │   │ synthetic_basic_002.csv
    │   │ synthetic_basic_003.csv
    │   │ ...
    │   
    └───synthetic_soil
    │   │ synthetic_soil_001.csv
    │   │ synthetic_soil_002.csv
    │   │ synthetic_soil_003.csv
    │   │ ...
    │   
    └───synthetic_soil_weather
    │   │ synthetic_soil_weather_001.csv
    │   │ synthetic_soil_weather_002.csv
    │   │ synthetic_soil_weather_003.csv
    │   │ ...
    │   
    └───synthetic_soil_weather_locations
    │   │ synthetic_soil_weather_001.csv
    │   │ synthetic_soil_weather_002.csv
    │   │ synthetic_soil_weather_003.csv
    │   │ ...
    │   
    └───synthetic_weather
        │ synthetic_weather_001.csv
        │ synthetic_weather_002.csv
        │ synthetic_weather_003.csv
        │ ...
```

TODO Option to only extract particular datasets

FIXME Location names are dropped during renaming

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

| Attribute               | Column name             |  Units  |
|-------------------------|:-----------------------:|--------:|
| Power                   | Power                   | kW      |
| Plane of array          | POA                     | W/m^2   |
| Ambient Temperatur      | Tamb                    | °C      |
| Wind                    | Wind                    | m/s     |
| Degradation rate / year | Degradation_rate_per_yr | year^-1 |
| Soiling                 | soiling                 | -       |
