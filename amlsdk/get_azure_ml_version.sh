echo $(pwd)
echo $(ls -l)

pip install --upgrade azureml-sdk

azure_ml_version_file='/workspace/amlsdk/azureml_sdk_version.txt'

echo "before checking Azureml!" > $azure_ml_version_file
echo "Azureml version file before checking:"
cat $azure_ml_version_file

python -c 'import azureml.core;print(azureml.core.VERSION)' > $azure_ml_version_file

echo "Azureml version file after checking:"
cat $azure_ml_version_file

crt_time=`date '+%Y_%m_%d_at_%H_%M_%S'`
azureml_sdk_v=`cat $azure_ml_version_file`
#echo $crt_time
#echo $azureml_sdk_v

cp $azure_ml_version_file /workspace/amlsdk/docker_history/azureml_sdk_version_is_"$azureml_sdk_v"_on_$crt_time.txt 

ls -l /workspace/amlsdk/docker_history/azureml_sdk_version_is_"$azureml_sdk_v"*



