### Copyright (C) Microsoft Corporation.   
  
  
# PowerAIWithDocker  
  
### Overview  
Reproducibility of Machine Learning (ML) model development and operationalization processes is a fundamental requirement for AI solutions in the cloud and on prem.    
    
ML application development steps cover 4 fundamental stages: 
 * experimentation orchestration (EO)  
 * running ML model training experiments (e13n)
 * ML scoring script development (SS)
 * operationalization (o16n).  
  
Each of these stages needs to be reproducible to ensure future auditing of the ML process and to allow future iterations of the ML model development.   
  
Docker containers provide the ideal solution for development of reproducible ML model training and operationalization. Furthermore, docker containers are the industry standard for deploying applications on prem and in the cloud, and in combination with Kubernetes Container Services they provide virtually out of the box scalability for enterprise applications.   
  
In general, containerization is easiest and best performed during the AI pipeline development process, by gradually extending the pre-processing steps and the complexity of the ML modeling process (like trying multiple ML algorithms or different deep learning frameworks like TF and pytorch) while keeping track of the used packages and their versions. In the end, the final development environment needs to be snapshot for later reproducibility.   
  
Azure Machine Learning Services (AML) Python SDK SDK allows one to address the ML model training step in a flexible way by starting with a generic base docker image like continuum conda/miniconda or nvidia for GPU processing, and then developing the script environment by altering the original default conda environment. It also provides a nice API for Azure AI resources like remote compute contexts on Ubuntu DSMs, and also model management capabilities.  
  
The other three ML app development stages (EO, SS) are not conceptually different than e13n and can also be based on docker images/ The o16n stage is fully covered by AML SDK, but docker is still critical for scoring script development before the Azure o16n flask app is created.  
   
### Design  
We provide an e2e workflow that shows how to create each of the above 4 docker images both within and outside AML Python SDK to create reproducible ML pipelines in Azure. We show both regular ML case (using simulated data and Kernel SVM to build a curved classification hyperplane) and deep learning using pretrained models for image classification using Keras/TF framework.  
  
We will use a Jupyter notebook running on the provisioned Azure DLVM to run the EO container (based on AML SDK) or to manually create the training docker image (outside SDK). Training docker image will run in a container on the same DLVM.  
  
We'll connect to it via a second Jupyter Notebook server, and we will develop the training script and train a deep learning model for image classification. The trained model and its associated scring script will then be deployed via a scoring docker image on an a [Azure Kubernetes Service (AKS)](https://azure.microsoft.com/en-us/services/kubernetes-service/) cluster.  
  
  
### Prerequisites:
 * Deploy an [Azure Deep Learning Virtual Machines (DLVM)](http://aka.ms/dlvm)
 * Open up ports for ssh, plus 2 Jupyter Notebook servers (one plain and the other one used for building the dockerized training and scoring scripts).  
   __NOTE__: this is __NOT__ a secure way to develop AI solutions. Securing access to VM and to the notebook server is paramount, but outside the scope of this tutorial. It is highly recommended to address the security issue before starting an AI development project.  
 * Add disks or expand the current ones as needed (you will need several 100 GB to store data and images). You can do this via portal or ps CLI:
```
# based on https://docs.microsoft.com/en-us/azure/machine-learning/preview/known-issues-and-troubleshooting-guide#vm-disk-is-full
#Deallocate VM (stopping will not work)  
$ az vm deallocate --resource-group myResourceGroup  --name myVM  
# Update Disc Size  
$ az disk update --resource-group myResourceGroup --name myVM --size-gb 250  
# Start VM     
$ az vm start --resource-group myResourceGroup  --name myVM  
``` 

 
* login (ssh) into the VM and create the project base directory structure:
```python
sudo mkdir -p /datadrive01
sudo chmod -R ugo=rwx  /datadrive01/
sudo mkdir -p /datadrive01/prj
sudo mkdir -p /datadrive01/data
sudo chmod -R ugo=rwx  /datadrive01/
```
* Login into dockerhub:
```
docker login
```
* [Get rid of sudo](https://docs.docker.com/install/linux/linux-postinstall/#manage-docker-as-a-non-root-user) in cli if you wish so:
```
sudo groupadd docker
sudo usermod -aG docker $USER
```
* Update/install a few system libs:
```
sudo apt-get update
pip install --upgrade pip
sudo apt-get install tmux
pip install -U python-dotenv
```
* [Fork](https://guides.github.com/activities/forking/) the [project](https://github.com/georgeAccnt-GH/PowerAIWithDocker.git) then clone it to your working computer
```
cd /datadrive01/prj/
git clone https://github.com/your_GitHub_account/PowerAIWithDocker.git
sudo chmod -R ugo=rwx  /datadrive01/
```
* The project code structure is shown below. 
 
### Setup:

### Cleaning up:

### Contributing:

 