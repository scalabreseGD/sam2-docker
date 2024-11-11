# SAM-2 Serving as Docker Rest Endpoint

## Before starting
<b>Build the model with GPU requires nvidia base image. Follow the instructions before in order to login into nvcr.io</b> 

<ol>
<li>Be sure you have Docker installed and buildkit available</li> 
<li>Connect to [NVIDIA NGC](https://org.ngc.nvidia.com/setup/personal-keys)</li>
<li>Create a personal key</li>
<li>Login into nvcr.io</li>
</ol>

## Build and Run
### Version
```shell
export version=x.y.z
```
### GPU
```shell
bash install.sh  sam-2 gpu registry
```

```shell
docker run --gpus all -p 8080:8080 -e PORT=8080 -v ./huggingface:/root/.cache/huggingface/ sam-2-gpu:${version}
```

### CPU
```shell
bash install.sh  sam-2 cpu registry
```
```shell
docker run -p 8080:8080 -e PORT=8080 -v ./huggingface:/root/.cache/huggingface/ sam-2-cpu:${version}
```
