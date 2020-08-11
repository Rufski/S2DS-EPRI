#!/bin/sh

# paths to the data zip-file and the git-destinationsitory
path_to_zip=$1 
path_to_destination=$2

# check if input parameters are sane
if [ "$#" -ne 2 ]; then
    echo "\nillegal number of parameters!\n"
    exit
fi

if [ ! -f "$path_to_zip" ]; then
    echo "\ninvalid path to data-file!\n"
    exit
fi

if [ ! -d "$path_to_destination" ]; then
    echo "\ninvalid path to destination directory!\n"
    exit
fi

# create data directory in the project if it does not exist
if [ -d "$path_to_destination/data/raw" ]; then
  printf "\ndirectory '${path_to_destination}data/raw' already exists\n"
else
  printf "\ncreating directory '${path_to_destination}data/raw'\n"
  mkdir "$path_to_destination/data"
fi

# extract data zip-file which contains the zip-files for individual datasets
printf "\nextracting datasets from ${path_to_zip}\n"
unzip -x "$path_to_zip" -d "$path_to_destination/data/raw/"

# move datasets
printf "\nmoving datasets from '${path_to_destination}data/raw/Data/' to '${path_to_destination}data/raw/'\n"
for zipfile_name in "Synthetic_Basic.zip" "Synthetic_Soil.zip" "Synthetic_Soil_Weather_Locations.zip" "Synthetic_Soil_and_Weather.zip" "Synthetic_Weather.zip"; do
    mv ${path_to_destination}data/raw/Data/${zipfile_name} $path_to_destination/data/raw/;
done

# remove files and dirs
printf "\nremoving file '${path_to_destination}data/raw/Data/data descriptions and notes.txt'\n"
rm "${path_to_destination}data/raw/Data/data descriptions and notes.txt"
printf "\nremoving directory '${path_to_destination}data/raw/Data/'\n"
rmdir "${path_to_destination}data/raw/Data/"
