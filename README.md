# Pytorch Model Splitting
This work is done in line with the <a href="https://github.com/gramineproject/gramine">Gramine-SGX</a> framework that ports unmodified applications inside SGX enclaves. The Gramine <a href="https://gramine.readthedocs.io/en/stable/tutorials/pytorch/index.html">documentation</a> consists of a **PyTorch PPML Framework Tutorial** that allows for privacy preservation when inferencing ML models. The PyTorch Model Splitting work enhances the security measurements of this framework by partitioning a model into smaller submodels to increase the confidentiality of the usage of the models. Inferencing can only be done when the submodels are arranged correctly.

This implementation can be used on the <a href="https://github.com/deislabs/mystikos">Mystikos</a> framework by porting over the python PyTorch inferencing application. Mystikos has a PyTorch/Onnx inference example but the documentation has since been removed. However, the other python application porting examples give a rough idea on how to execute the private inference <a href="https://github.com/deislabs/mystikos/blob/main/doc/user-getting-started-docker-python.md">here</a>. The instructions to run the Mystikos implementation will not be elaborated here. The ```Mystikos/``` dir contains the ```dockerfile``` and a shell script to launch the Mystikos application.

# Instructions
* Clone this repo: `git clone https://github.com/haadeescott/pytorch_ModelSplitting `
* Each Model folder contains 2 subfolders that has a base model inference, and a model that has been split into 12 parts to conduct an inference.
   * eg: `Alexnet/12 submodels/pyTorchSplit_alexnet12.ipynb`
   * Uncomment and execute the **save pytorch model** cell to save the models inside the folder
* The `<..>.py` file is the program to run the models inside the Gramine-SGX enclave
   * Requires the `pytorch.manifest.template` file to be updated with the appropriate files inside the **trusted** and **allowed** blocks. Example can be seen below.
* Once Model(s) has been downloaded and saved using the **pyTorchSplit_<model>.ipynb** file, you can encrypt the Model(s) using the **encryptModel.py** program with any of the secret keys in the ```secrets/``` dir, and include ```<model>_enc.pt``` as its uniformed name. 
* Upon decryption, you may remove the ```<model>_enc.pt``` and replace it as ```<model>.pt``` as can be seen as the standardised naming convention in the manifest file below. 


## To note:
> Only the <a href="https://github.com/haadeescott/pytorch_ModelSplitting/blob/main/Alexnet/12%20submodels/pyTorchSplit_Alexnet_12.py">AlexNet</a> folder has the python inference script to decrypt 12 submodels and inference. You may replace this with other models. Most split submodels have the same naming convention, but just check again and ensure all submodel names are encrypted, decrypted, inferenced accordingly.


## VM System information
- OS: Linux (Ubuntu 20.0.04 LTS)
- Size: Standard Dc4s v2 (4vcpus, 16GiB memory)
- CPU Processors: 4
- CPU Model: Intel(R) Xeon(R) E-2288G CPU @ 3.70GHz
- Kernel Version: Linux Version 5.13.0-1017-Azure
- Allocated size for Gramine-SGX enclave: 8GB


## Video demonstration on system process flow
<a href="https://youtu.be/Zuak5Wn50jA" target="_blank">
<img src="https://github.com/haadeescott/pytorch_ModelSplitting/blob/main/Results/plot_Images/details_video.png" height="300" width="540">
</a>

**The objective is to ensure the user is able to execute inferencing while preserving privacy of their computation and their model.**
Firstly encrypt the model(s) using a Key. Encode the Key with a wrapper-key => Secret Key with the SGX protected file mechanism. The Secret Key now can only be decoded when the user completes the EPID Attestation. where the Intel attestation service will provision a secret that decodes and reads the Secret Key into a useable Key. Once completed, the user can then launch the SGX enclave as per the configurations from the SGX manifest file and decrypt the model, and use Gramine-SGX pytorch inferencing function to execute the script.

### Disclaimer
> The demonstration includes the attestation service that is not covered by this repository. The only works included here are the model splitting process and the benchmarking performance tests


## Prepare all the pretrained models
Run `python3 dl-pretrained-models.py` to download and save all the pretrained models:
```  
   - Alexnet
   - MobilenetV3 Large
   - MobilenetV3 Small
   - Resnet50
   - Resnet101
   - Squeezenet
   - Vgg16
   - Vgg19 
```
and store them inside the designated folders.

### Concept for Model Splitting
<img src="https://github.com/haadeescott/pytorch_ModelSplitting/blob/main/Results/plot_Images/Diagram_4_Comparisons.png"></br>
The model is split/partitioned after iterating through each layer of the model that can be accessed and extracted.

## Guide for Model Splitting
- Requires torchvision to be installed: `` pip3 install torchvision``
- All the model networks has an `AdaptiveAvgPool2d` layer that allows the output to be `Flatten()` and feed into the next layer to complete the inferencing after splitting the model. Without flattening, after splitting the model, the execution of model inference cannot be completed. 
- The extraction of layers for each submodels can be adjusted accordingly, however the `<pyTorchSplit_...>.ipynb` can serve as a template.
- Once all the **model/submodels** have been saved and dowloaded, store them into the appropriate folders to execute inference.

## Prepare environment for inferencing in Gramine-SGX enclave
Follow the instructions and install the SGX SDK and drivers [here](https://github.com/intel/linux-sgx.git). The `pytorch.manifest.template` requires the files to be appended and updated before compiling the program. Change the `sgx.enclave_size = "8G"` to the desired size **(recommended 4G and above)**. In the `sgx.trusted_files` block, include the python program, model, label class text file, and the input image to be inferenced. The `sgx.allowed_files` block should include the results text file for the output.

```

sgx.nonpie_binary = true
sgx.enclave_size = "8G"
sgx.thread_num = 256

sgx.trusted_files = [
  "file:{{ gramine.libos }}",
  "file:{{ entrypoint }}",
  "file:{{ gramine.runtimedir() }}/",
  "file:{{ arch_libdir }}/",
  "file:/usr/{{ arch_libdir }}/",
  "file:{{ python.stdlib }}/",
  "file:{{ python.distlib }}/",
  "file:{{ env.HOME }}/.local/lib/",
  "file:{{ python.get_path('stdlib', vars={'installed_base': '/usr/local'}) }}/",
  "file:pyTorchSplit_alexnet.py", // include python script to run here
  "file:classes.txt",
  "file:input.jpg",

  # Pre-trained model saved as a file
  "file:alexnet-pretrained.pt"
  # Uncomment line below if you want to use torchvision.model.alexnet(pretrained=True)
  # "file:{{ env.HOME }}/.cache/torch/checkpoints/alexnet-owt-4df8aa71.pth",
]

sgx.allowed_files = [
  "file:/tmp",
  "file:/etc/apt/apt.conf.d",
  "file:/etc/apt/apt.conf",
  "file:/etc/default/apport",
  "file:/etc/nsswitch.conf",
  "file:/etc/group",
  "file:/etc/passwd",
  "file:/etc/host.conf",
  "file:/etc/hosts",
  "file:/etc/gai.conf",
  "file:/etc/resolv.conf",
  "file:/etc/fstab",
  "file:result.txt",
  "file:alexnet-pretrained.pt", // include name(s) of unencrypted model
  "file:alexnet-pretrained_enc.pt", // include name(s) of encrypted model
  "file:secrets/", // include secrets dir to allow secret keys to be read
]
```
- Once the `pytorch.manifest.template` file has been updated, compile with SGX and execute the inference: 
  - **gramine-sgx ./pytorch <python_script>.py**
```
make SGX=1
gramine-sgx ./pytorch pyTorchSplit_alexnet.py
```

## Results
The results folder contains all the `.csv` files that has the performance results and metrics for each of the model networks. `../plot_Images/` contains all the line charts plotted with the metrics and the benchmarking performances. The performance tests include:
- CPU Utilisation %
- Memory Footprint KB
- Power Consumption W

The performance of a single execution were broken down into different segments:
- overhead incurred
- inference
- total execution time
These were benchmarked against both native and SGX environment to observe the overhead costs.

#### Gramine-SGX against Mystikos-SGX
Mystikos-SGX consumes higher amounts of memory due to its usage of docker. </br>
<img src="https://github.com/haadeescott/pytorch_ModelSplitting/blob/main/Results/plot_Images/Performance_Against_Mystikos.png" width="300" height="150">
<img src="https://github.com/haadeescott/pytorch_ModelSplitting/blob/main/Results/plot_Images/Performance_Against_Mystikos_2.png" width="300" height="150">

### Gramine-SGX AlexNet performance
The Alexnet model is benchmarked against the different number of partitions for the inference timing. The Alexnet model is measured against 12 submodels of Alexnet partitioned, and its performance is plotted to identify if any overhead has incurred. </br>
<img src="https://github.com/haadeescott/pytorch_ModelSplitting/blob/main/Results/plot_Images/Performance_Alexnet_All_Submodels.png">

# Tests
The script ```Alexnet/pyTorchSplit_alexnet_test.py``` has a 2nd inferencing function that aims to identify the isolated inference time of running said execution in an SGX environment, as SGX does not allow any application based profiling inside the enclave. The performance metrics can be executed by running the ```readMetrics.sh``` shell script which executes reading of memory footprint, cpu utilisation, and power consumption over 100 iterations of 1 second cycle frame and 10 iterations of 10 seconds cycle frame, respectively.

