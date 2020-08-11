#!/bin/bash

# paths to folder containing csv
path_to_folder=$1 
filename_prefix=$(echo $(basename ${path_to_folder}))

# check if input parameters are sane
if [ "$#" -ne 1 ]; then
    echo "\nillegal number of parameters!\n"
    exit
fi

if [ ! -d "$path_to_folder" ]; then
    echo "\ninvalid path to repo directory!\n"
    exit
fi

# get old filename prefix
if [ "$filename_prefix" == "synthetic_basic" ]; then
  filename_prefix_old='Synthetic_Basic';
elif [ "$filename_prefix" == "synthetic_soil" ]; then
  filename_prefix_old='Synthetic_Soil';
elif [ "$filename_prefix" == "synthetic_weather" ]; then
  filename_prefix_old='Synthetic_Weather';
elif [ "$filename_prefix" == "synthetic_soil_weather" ]; then
  filename_prefix_old='Synthetic_Soil_and_Weather';
elif [ "$filename_prefix" == "synthetic_soil_weather_locations" ]; then
  filename_prefix_old='Synthetic_Soil_Weather_Locations';
else
  echo "\ninvalid dataset directory!\n";
  exit;
fi


echo $filename_prefix
echo $filename_prefix_old

#echo "${path_to_folder}${filename_prefix}_"
#echo $(pwd)
#echo $(ls)




cd ${path_to_folder}




# fix "and" problem for synthetic soil "and" weather
if [ "$filename_prefix" == "synthetic_soil_weather" ]; then
  printf "\nfix 'and'-problem of 'Synthetic_Soil_and_Weather'\n" 
  find . -type f -name "${filename_prefix_old}*" -print0 | while read -d $'\0' f
  do
     new=`echo "$f" | sed -e "s/${filename_prefix_old}/${filename_prefix}/"`
     mv "$f" "$new"
  done
fi

# rename uppercase to lowercase
printf "\nrename uppercase to lowercase\n" 
for f in `ls`; do mv -n "$f" "`echo $f | tr '[A-Z]' '[a-z]'`"; done

# no padding zeros for now
