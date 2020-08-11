#!/bin/bash

# path to the dataset zip-file
path_to_zip=$1 

# check if input parameters are sane
if [ "$#" -ne 1 ]; then
    echo "\nillegal number of parameters!\n"
    exit
fi

if [ ! -f "$path_to_zip" ]; then
    echo "\ninvalid path to data-file!\n"
    exit
fi


# check which zip-file it is
filename_zip=$(basename $path_to_zip)
if [ "$filename_zip" == "Synthetic_Basic.zip" ]; then
  data_prefix_new='synthetic_basic';
elif [ "$filename_zip" == "Synthetic_Soil.zip" ]; then
  data_prefix_new='synthetic_soil';
elif [ "$filename_zip" == "Synthetic_Soil_Weather_Locations.zip" ]; then
  data_prefix_new='synthetic_soil_weather_locations';
elif [ "$filename_zip" == "Synthetic_Soil_and_Weather.zip" ]; then
  data_prefix_new='synthetic_soil_weather';
elif [ "$filename_zip" == "Synthetic_Weather.zip" ]; then
  data_prefix_new='synthetic_weather';
else
  echo "\ninvalid zip-file name!\n";
  exit;
fi

# create directory and extract dataset
printf "\ncreating directory '$(dirname ${path_to_zip})/${data_prefix_new}'\n"
mkdir "$(dirname ${path_to_zip})/${data_prefix_new}"
printf "\nextracting '${path_to_zip}' to '$(dirname ${path_to_zip})/${data_prefix_new}'\n"
unzip -x "${path_to_zip}" -d "$(dirname ${path_to_zip})/${data_prefix_new}"



# soil and weather data somehow have an additional folder depth
if [ "$filename_zip" == "Synthetic_Soil_and_Weather.zip" ]; then
    printf "\nfixing problem of additional directory-depth for 'Synthetic_Soil_and_Weather.zip'\n"
    mv $(dirname ${path_to_zip})/${data_prefix_new}/Synthetic_Soil_and_Weather/* $(dirname ${path_to_zip})/${data_prefix_new};
    rmdir $(dirname ${path_to_zip})/${data_prefix_new}/Synthetic_Soil_and_Weather/;
fi
