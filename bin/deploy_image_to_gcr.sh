
#local_name=INSERT_LOCAL_CONTAINER_NAME
#hostname=us.gcr.io
#target_name=$local_name
#project_id=quick-redraw-267111
#
#docker tag $local_name $hostname/$project_id/


# Or, do this remote (WAY faster, since I dont have to upload the build images)
# gcloud builds submit --tag us.gcr.io/quick-redraw-267111/test_img_remotebuild