to_download=( cat cow )

for item in "${to_download[@]}" ; do
  gsutil -m cp gs://quickdraw_dataset/full/simplified/$item.ndjson .
done