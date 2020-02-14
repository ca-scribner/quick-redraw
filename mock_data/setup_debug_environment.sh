#!/bin/bash

readarray -t to_download  < ./drawings_to_download
destination=untracked
extension=bin
db=metadata.sqlite
max_drawings=1000

./download_binary_drawings.sh

# update pythonpath so we have this package available
package_dir=`pwd`/..
export PYTHONPATH="${PYTHONPATH}:${package_dir}"
echo $PYTHONPATH

# populate db with raw and normalized drawings
cd $destination
for item in "${to_download[@]}" ; do
  echo "Adding $item to $db"
  python ../load_data.py $item.$extension $item $db ./raw/ --max_drawings $max_drawings --normalized_storage_location ./normalized/
done
cd -
