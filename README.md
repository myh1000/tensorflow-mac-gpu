# tensorflow-mac-gpu

Instructions based off of  [Mistobaan](https://gist.github.com/Mistobaan)'s [gist](https://gist.github.com/Mistobaan/dd32287eeb6859c6668d).

These instructions are what I did to install tensorflow on my machine after a segmentation error occured while trying to import tensorflow in python took me a whole lot of debugging to fix.

## Prerequisites

Update ```brew``` and install latest packages.
```
brew update
brew install coreutils swig bazel
```
Now you can either install CUDA via ```brew``` or get it from NVIDIA themselves [here.](https://developer.nvidia.com/cuda-downloads)

Check to make sure the version is above ```7.5```
```
brew cask info cuda
```
```
cuda: 7.5.27
Nvidia CUDA
https://developer.nvidia.com/cuda-zone
/usr/local/Caskroom/cuda/7.5.27 (23 files, 1.0G)
https://github.com/caskroom/homebrew-cask/blob/master/Casks/cuda.rb
No Artifact Info
```
Now you need to get NVIDIA's cuDNN library. You will have to register and download it from their website: https://developer.nvidia.com/cudnn.

I personally used cuDNN-6.5 for v2, becuase thats what worked, but you might find that v5 works better for you.

Once downloaded you need to manually copy the files over the ```/usr/local/cuda/``` directory.
```
tar xzvf ~/Downloads/cudnn-6.5-osx-x64-v2.0-rc.tgz
sudo mv -v cuda/lib/libcudnn* /usr/local/cuda/lib
sudo mv -v cuda/include/cudnn.h /usr/local/cuda/include
```

add in your ```~/.bash_profile``` the reference to ```/usr/local/cuda/lib```. You will need it in order for tensorflow to find the libraries.
```
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda/lib"
```
Others have instead exported ```DYLD_LIBRARY_PATH``` instead of ```LD_LIBRARY_PATH```, so if the installation fails or there are problems importing tensorflow, try changing this.

After that, reload the bash_profile ```. ~/.bash_profile``` or just close and reopen the terminal to apply the change.

Now you will need to find the Compute Capability for your graphics card. One way to do it is to find the exact model NVIDIA card you have and look up the model on the list here: https://developer.nvidia.com/cuda-gpus.

Another way is to compile the deviceQuery utility found inside the cuda sdk repository.
```
cd /usr/local/cuda/samples
sudo make -C 1_Utilities/deviceQuery
./bin/x86_64/darwin/release/deviceQuery
```

The output should be something like:
```
UDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 775M"
  CUDA Driver Version / Runtime Version          7.5 / 7.5
  CUDA Capability Major/Minor version number:    3.0
  Total amount of global memory:                 2048 MBytes (2147024896 bytes)
  ( 7) Multiprocessors, (192) CUDA Cores/MP:     1344 CUDA Cores
  GPU Max Clock rate:                            797 MHz (0.80 GHz)
  Memory Clock rate:                             2500 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 524288 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 7.5, CUDA Runtime Version = 7.5, NumDevs = 1, Device0 = GeForce GTX 775M
Result = PASS
```
Here you can confirm that the driver is set to 7.5 (```7.5 / 7.5```) and you can find also the cuda capability of your GPU, ```CUDA Capability Major/Minor version number: 3.0``` in my case, you will need this in order to properly configure tensorflow.

## Clone Tensorflow

You will need to clone the repository.
```
git clone --recurse-submodules https://github.com/tensorflow/tensorflow
cd tensorflow
git checkout master
```
Then you can configure it, making sure to fill out which versions of CUDA and cuDNN you are using. 
```
./configure
Please specify the location of python. [Default is /usr/bin/python]:
Do you wish to build TensorFlow with GPU support? [y/N] y
GPU support will be enabled for TensorFlow

Please specify which gcc nvcc should use as the host compiler. [Default is
/usr/bin/gcc]:

Please specify the Cuda SDK version you want to use, e.g. 7.0. [Leave
empty to use system default]: 7.5

Please specify the location where CUDA 7.5 toolkit is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]: /usr/local/cuda

Please specify the Cudnn version you want to use. [Leave empty to use system
default]: 6.5

Please specify the location where the cuDNN 6.5 library is installed. Refer to
README.md for more details. [default is: /usr/local/cuda]:

Please specify a list of comma-separated Cuda compute capabilities you want to
build with. You can find the compute capability of your device at:
https://developer.nvidia.com/cuda-gpus.
Please note that each additional compute capability significantly increases your
build time and binary size. [Default is: \"3.5,5.2\"]: 3.0

Setting up Cuda include
Setting up Cuda lib64
Setting up Cuda bin
Setting up Cuda nvvm
Setting up CUPTI include
Setting up CUPTI lib64
Configuration finished
```
Now we can build the tensorflow pip package. This will take awhile.
```
bazel build -c opt --config=cuda //tensorflow/cc:tutorials_example_trainer
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow-0.9.0-py2-none-any.whl
```

## Verify Installation

You need to exit the tensorflow directory ```cd ~``` in order to test your installation or all you will recieve is:
```
ImportError: cannot import name 'pywrap_tensorflow'
```
Now run ```python``` and run a test script.
```
import tensorflow as tf

# Creates a graph.
a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)

# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

# Runs the op.
print sess.run(c)
```
You should get something like:
```
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.7.5.dylib locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcudnn.6.5.dylib locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcufft.7.5.dylib locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcuda.1.dylib locally
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcurand.7.5.dylib locally
I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:892] OS X does not support NUMA - returning NUMA node zero
I tensorflow/core/common_runtime/gpu/gpu_init.cc:102] Found device 0 with properties:
name: GeForce GTX 775M
major: 3 minor: 0 memoryClockRate (GHz) 0.797
pciBusID 0000:01:00.0
Total memory: 2.00GiB
Free memory: 454.24MiB
I tensorflow/core/common_runtime/gpu/gpu_init.cc:126] DMA: 0
I tensorflow/core/common_runtime/gpu/gpu_init.cc:136] 0:   Y
I tensorflow/core/common_runtime/gpu/gpu_device.cc:839] Creating TensorFlow device (/gpu:0) -> (device: 0, name: GeForce GTX 775M, pci bus id: 0000:01:00.0)
Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GeForce GTX 775M, pci bus id: 0000:01:00.0
I tensorflow/core/common_runtime/direct_session.cc:175] Device mapping:
/job:localhost/replica:0/task:0/gpu:0 -> device: 0, name: GeForce GTX 775M, pci bus id: 0000:01:00.0

MatMul: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:818] MatMul: /job:localhost/replica:0/task:0/gpu:0
b: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:818] b: /job:localhost/replica:0/task:0/gpu:0
a: /job:localhost/replica:0/task:0/gpu:0
I tensorflow/core/common_runtime/simple_placer.cc:818] a: /job:localhost/replica:0/task:0/gpu:0
[[ 22.  28.]
 [ 49.  64.]]
 ```

 ## Possible Errors

 If you don't see your error [here](https://gist.github.com/Mistobaan/dd32287eeb6859c6668d#caveats), then read on for the errors I had.

 ```
 ImportError: cannot import name 'pywrap_tensorflow'
 ```
 You are running the script from the same tensorflow directory and python is using the local directory as the module. Change directory ```cd ~```

```
>>> import tensorflow
I tensorflow/stream_executor/dso_loader.cc:108] successfully opened CUDA library libcublas.7.5.dylib locally
Segmentation fault: 11
```
This error most likely occurred because tensorflow was unable to find the CUDA library. Make sure to set the environment variable.
```
export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
```
If you did this earlier, and you get this error, try replacing ```LD_LIBRARY_PATH``` with ```DYLD_LIBRARY_PATH```.
If this still didn''t work, perhaps you can try your luck here: [issue #2773](https://github.com/tensorflow/tensorflow/issues/2773).
