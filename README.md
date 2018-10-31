### Copyright (C) Microsoft Corporation.   
  
  
# PowerAIWithDocker  
  
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
   
We provide an e2e workflow that shows how to create each of the above 4 docker images both within and outside AML Python SDK to create reproducible ML pipelines in Azure. We show both regular ML case (using simulated data and Kernel SVM to build a curved classification hyperplane) and deep learning using pretrained models for image classification using Keras/TF framework.  
