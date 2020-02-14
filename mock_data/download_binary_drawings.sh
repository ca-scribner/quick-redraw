#!/bin/bash

readarray -t to_download  < ./drawings_to_download
extension=bin
destination=untracked

mkdir -p $destination/raw
mkdir -p $destination/normalized

for item in "${to_download[@]}" ; do
  item=$item.$extension
  if [ ! -f ./$destination/$item ]; then
    echo "Downloading $item"
    gsutil -m cp gs://quickdraw_dataset/full/binary/$item ./$destination/$item
  else
    echo "$item already exists - skipping"
  fi
done
