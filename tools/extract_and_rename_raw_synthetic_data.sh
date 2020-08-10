#!/bin/sh

# paths to the data zip-file and the git-repository
path_to_zip=$1 
path_to_repo=$2

# check if input parameters are sane
if [ "$#" -ne 2 ]; then
    echo "illegal number of parameters!"
    exit
fi

if [ ! -f "$path_to_zip" ]; then
    echo "invalid path to data-file!"
    exit
fi

if [ ! -d "$path_to_repo" ]; then
    echo "invalid path to repo directory!"
    exit
fi

# function to extract the different datasets
create_dataset_dir () {
  dirname_dataset=$1
  printf "\ncreating directory '/data/raw/$dirname_dataset'\n"
  mkdir "$path_to_repo/data/raw/$dirname_dataset"
}

# function to extract the different datasets and rename files and dirs
extract_dataset_and_rename () {
  dataset_prefix_old=$1
  datafile_prefix_new=$2
  printf "unzipping datasets to '/data/raw/${datafile_prefix_new}'\n\n"
  unzip -x "${path_to_repo}/data/raw/Data/${dataset_prefix_old}.zip" -d "${path_to_repo}/data/raw/${datafile_prefix_new}/"
  printf "\nrenaming datafiles in '/data/raw/${datafile_prefix_new}/'\n\n"
  cd "${path_to_repo}/data/raw/${datafile_prefix_new}/"
  for f in `ls`; do mv -n "$f" "`echo $f | tr '[A-Z]' '[a-z]'`"; done
  ls * | cat -n | while read i f; do mv -n "$f" `printf "${datafile_prefix_new}_%03d.${f#*.}" "$i"`; done
}

# create data directory in the project if it does not exist
if [ -d "$path_to_repo/data/raw" ]; then
  printf "\ndirectory '/data/raw' already exists\n\n"
else
  printf "\ncreating directory '/data/raw'\n\n"
  mkdir "$path_to_repo/data"
fi

# extract data zip-file which contains the zip-files for individual datasets
printf "unzipping datafile\n\n"
unzip -x "$path_to_zip" -d "$path_to_repo/data/raw/"

# extract different datasets, remove capitalization in filenames and pad numbers with zeros
create_dataset_dir "synthetic_basic"
extract_dataset_and_rename "Synthetic_Basic" "synthetic_basic"
create_dataset_dir "synthetic_soil"
extract_dataset_and_rename "Synthetic_Soil" "synthetic_soil"
create_dataset_dir "synthetic_weather"
extract_dataset_and_rename "Synthetic_Weather" "synthetic_weather"
create_dataset_dir "synthetic_soil_weather"
extract_dataset_and_rename "Synthetic_Soil_and_Weather" "synthetic_soil_weather"
create_dataset_dir "synthetic_soil_weather_locations"
extract_dataset_and_rename "Synthetic_Soil_Weather_Locations" "synthetic_soil_weather_locations"

# remove extracted data directory
rm -r "${path_to_repo}/data/raw/Data/"
