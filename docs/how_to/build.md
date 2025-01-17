
Installation Guide
==================

This page gives instructions of how to build and install the mxnet package from
scratch on various systems. It consists of two steps, first we build the shared
library from the C++ codes (`libmxnet.so` for linux/osx and `libmxnet.dll` for
windows). Then we install the language, e.g. Python, packages. If the
instructions on this page do not work for you, please feel free to ask questions
at [mxnet/issues](https://github.com/dmlc/mxnet/issues), or even better to send
pull request if you can fix the problem.

## Contents
- [Build the Shared Library](#build-mxnet-library)
  - [Prerequisites](#prerequisites)
  - [Building on Ubuntu/Debian](#building-on-ubuntu-debian)
  - [Building on OSX](#building-on-osx)
  - [Building on Windows](#building-on-windows)
  - [Installing pre-built packages on Windows](#installing-pre-built-packages-on-windows)
  - [Customized Building](#customized-building)
- [Python Package Installation](#python-package-installation)
- [R Package Installation](#r-package-installation)
- [Julia Package Installation](#julia-package-installation)
- [Docker Images](#docker-images)

## Build the Shared Library

Our goal is to build the shared library:
- On Linux/OSX the target library is ```libmxnet.so```
- On Windows the target libary is ```libmxnet.dll```

The minimal building requirement is

- A recent c++ compiler supporting C++ 11 such as `g++ >= 4.8` or `clang`
- A BLAS library, such as `libblas`, `libblas`, `openblas` `intel mkl`

Optional libraries

- `CUDA Toolkit >= v7.0` to run on nvidia GPUs
  - Requires GPU with support for `Compute Capability >= 2.0`
- CUDNN to accelerate the GPU computation
- opencv for image augmentation

We can edit `make/config.mk` to change the compile options, and then build by
`make`. If everything goes well, we can go the
[language package installation](#install-language-packages) step.

On the remaining of this section, we provide instructions to install the
dependencies and build mxnet from scratch for various systems.

### Building on Ubuntu/Debian

On Ubuntu >= 13.10, one can install the dependencies by

```bash
sudo apt-get update
sudo apt-get install -y build-essential git libatlas-base-dev libopencv-dev
```

Then build mxnet
```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; make -j$(nproc)
```

### Building on OSX

On OSX, we can install the dependencies by

```bash
brew update
brew tap homebrew/science
brew info opencv
brew install opencv
```

Then build mxnet

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; cp make/osx.mk ./config.mk; make -j$(sysctl -n hw.ncpu)
```

Or use cmake command and Xcode

```bash
mkdir build; cd build
cmake -G Xcode -DCMAKE_BUILD_TYPE=Release -DCMAKE_CONFIGURATION_TYPES="Release" -DUSE_OPENMP="OFF" -DUSE_CUDNN="OFF" -DUSE_CUDA="OFF" -DBLAS=MKL ..
```

Then open `mxnet.xcodeproj` by xcode and change two flags in `Build Settings` before building:
(1) Link-Time Optimization = Yes
(2) Optimisation Level = Fasteset[-O3]


Troubleshooting:

Some of the users might meet the link error `ld: library not found for -lgomp`, indicating that the GNU implementation of OpenMP is not in the library path of operating system.

To resolve this issue, run the following commands:

```
sudo launchctl load -w /System/Library/LaunchDaemons/com.apple.locate.plist # this creates the locate database if it does not exist

locate libgomp.dylib #copy the path which is generated by this command, say path1

ln -s path1 /usr/local/lib/libgomp.dylib

```

then run `make -j$(sysctl -n hw.ncpu)` again.


### Building on Windows

Firstly, we should make your Visual Studio 2013 support more C++11 features.

 - Download and install [Visual C++ Compiler Nov 2013 CTP](http://www.microsoft.com/en-us/download/details.aspx?id=41151).
 - Copy all files in `C:\Program Files (x86)\Microsoft Visual C++ Compiler Nov 2013 CTP` (or the folder where you extracted the zip archive) to `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC` and overwrite all existed files. Don't forget to backup the original files before copying.

Secondly, fetch the third-party libraries, including [OpenCV](http://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.0.0/opencv-3.0.0.exe/download), [CuDNN](https://developer.nvidia.com/cudnn) and [OpenBlas](http://sourceforge.net/projects/openblas/files/v0.2.14/)(ignore this if you have MKL).

 - NOTICE: You need to register as a NVIDIA community user to get the download link of CuDNN.

Finally, use CMake to create a Visual Studio solution in `./build/`. During configuration, you may need to set the path of each third-party library, until no error is reported. (Set environmental variable OpenBLAS_HOME to the OpenBLAS directory containing `include` and `lib`; set OpenCV_DIR to the `build` directory after unpacking the OpenCV package.) Open the solution and compile, you will get a `mxnet.dll` in `./build/Release` or `./build/Debug`.

### Installing pre-built packages on Windows

Mxnet also provides pre-built packages on Windows. The pre-built package includes pre-build MxNet library, the dependent thrid-party libraries, a sample C++ solution in Visual Studio and the Python install script.

You can download the packages from the [Releases tab](https://github.com/dmlc/mxnet/releases) of MxNet. There are two variants provided: one with GPU support (using CUDA and CUDNN v3) and one without GPU support. You can choose one that fits your hardward configuration.

After download, unpack the package into a folder, say D:\MxNet, then install the package by double clicking the setupenv.cmd inside the folder. It will setup environmental variables needed by MxNet. After that, you should be able to usee the provided VS solution to build C++ programs, or to [install Python package](#python-package-installation).

### Customized Building

The configuration of mxnet can be modified by ```config.mk```
- modify the compiling options such as compilers, CUDA, CUDNN, Intel MKL,
various distributed filesystem such as HDFS/Amazon S3/...
- First copy [make/config.mk](../../make/config.mk) to the project root, on which
  any local modification will be ignored by git, then modify the according flags.

#### Building with Intel MKL Support
First, `source /path/to/intel/bin/compilervars.sh` to automatically set environment variables. Then, edit [make/config.mk](../../make/config.mk), let `USE_BLAS = mkl`. `USE_INTEL_PATH = NONE` is usually not necessary to be modified.

#### Building for distributed training
To be able to run distributed training jobs, the `USE_DIST_KVSTORE=1` flag must be set.  This enables a distributed
key-value store needed to share parameters between multiple machines working on training the same neural network.

## Python Package Installation

The python package is located at [mxnet/python](../../python/mxnet). It requires
`python>=2.7` and `numpy`. To install the latter, if `pip` is available, then

```bash
sudo pip install numpy
```

otherwise use your package manager, e.g.

```bash
sudo apt-get install python-numpy # for debian
sudo yum install python-numpy # for redhat
```

To have a quick test of the python package, we can train
a MLP on the mnist dataset:

```bash
python example/image-classification/train_mnist.py
```

or train a convolution neural network using GPU 0 if we set `USE_CUDA=1` during
compiling:

```bash
python example/image-classification/train_mnist.py --network lenet --gpus 0
```

There are several ways to install the package:

1. Install system-widely, which requires root permission

   ```bash
   cd python; sudo python setup.py install
   ```

   You will however need Python `distutils` module for this to
   work. It is often part of the core python package or it can be installed using your
   package manager, e.g. in Debian use

   ```bash
   sudo apt-get install python-setuptools
   ```
2. Only set the environment variable `PYTHONPATH` to tell python where to find
   the library. For example, assume we cloned `mxnet` on the home directory
   `~`. then we can added the following line in `~/.bashrc`
   It is ***recommended for developers*** who may change the codes. The changes will be immediately reflected once you pulled the code and rebuild the project (no need to call ```setup``` again)

    ```bash
    export PYTHONPATH=~/mxnet/python
    ```

3. Install only for the current user.

    ```bash
    cd python; python setup.py develop --user
    ```

4. Copy the package into the working directory which contains the mxnet
   application programs. In this approach we don't need to change the system,
   and therefore is recommended for distributed training.

   Assume we are on the working directory, and `mxnet` is cloned on the home
   directory `~`.

   ```bash
   cp -r ~/mxnet/python/mxnet .
   cp ~/mxnet/lib/libmxnet.so mxnet/
   ```

## R Package Installation

For Windows/Mac users, we provide pre-built binary package using CPU.
You can install weekly updated package directly in R console:

```r
install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("mxnet")
```

To install the R package. First finish the [Build MXNet Library](#build-mxnet-library) step.
Then use the following command to install dependencies and build the package at root folder

```bash
Rscript -e "install.packages('devtools', repo = 'https://cran.rstudio.com')"
cd R-package
Rscript -e "library(devtools); library(methods); options(repos=c(CRAN='https://cran.rstudio.com')); install_deps(dependencies = TRUE)"
cd ..
make rpkg
```

Now you should have the R package as a tar.gz file and you can install it as a normal package by (the version number might be different)

```bash
R CMD INSTALL mxnet_0.5.tar.gz
```

If you can't load `mxnet` after enabling CUDA during the installation. Please add following lines into `$RHOME/etc/ldpaths`. You can find your `$RHOME` by using `R.home()` inside R.

```bash
export CUDA_HOME=/usr/local/cuda 
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
```

To install the package using GPU on Windows without building the package from scratch. Note that you need a couple of programs installed already:  
- You'll need the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). This depends on Visual Studio, and a free compatible version would be [Visual Studio Community 2013](https://www.visualstudio.com/en-us/news/vs2013-community-vs.aspx). For instructions and compatibility checks, read http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-microsoft-windows/ .

- You will also need to register as a developer at nvidia and download CUDNN V3, https://developer.nvidia.com/cudnn . 


1. Download the mxnet package as a ZIP from the Github repository https://github.com/dmlc/mxnet and unpack it. You will be editing the `/mxnet/R-package` folder.

2. Download the most recent GPU-enabled package from the [Releases tab](https://github.com/dmlc/mxnet/releases). Unzip this file so you have a folder `/nocudnn`. Note that this file and the folder you'll save it in will be used for future reference and not directly for installing the package. Only some files will be copied from it into the `R-package` folder.

(Note: you now have 2 folders we're working with, possibly in different locations, that we'll reference with `R-package/` and `nocudnn/`.)

3. Download CUDNN V3 from https://developer.nvidia.com/cudnn. Unpack the .zip file and you'll see 3 folders, `/bin`, `/include`, `/lib`. Copy and replace these 3 folders into `nocudnn/3rdparty/cudnn/`, or unpack the .zip file there directly.

4. Create the folder `R-package/inst/libs/x64`. We only support 64-bit operating system now, so you need the x64 folder;

5. Put dll files in `R-package/inst/libs/x64`. 

The first dll file you need is `nocudnn/lib/libmxnet.dll`. The other dll files you need are the ones in all 4 subfolders of `nocudnn/3rdparty/`, for the `cudnn` and `openblas` you'll need to look in the `/bin` folders. There should be 11 dll files now in `R-package/inst/libs/x64`.

6. Copy the folder `nocudnn/include/` to `R-package/inst/`. So now you should have a folder `R-package/inst/include/` with 3 subfolders.

7. Run `R CMD INSTALL --no-multiarch R-package`. Make sure that R is added to your PATH in Environment Variables. Running the command `Where R` in Command Prompt should return the location.

Note on Library Build:

We isolate the library build with Rcpp end to maximize the portability
  - MSVC is needed on windows to build the mxnet library, because of CUDA compatiblity issue of toolchains.

## Julia Package Installation

The Julia package is hosted in a separate repository [MXNet.jl](https://github.com/dmlc/MXNet.jl). To use the Julia binding with an existing libmxnet installation, set the following environment variable

```bash
export MXNET_HOME=/path/to/libmxnet
```

The path should be the root directory of libmxnet, in other words, `libmxnet.so` should be found at `$MXNET_HOME/lib`. You might want to add it to your `.bashrc`. Then the Julia package could be installed via

```julia
Pkg.add("MXNet")
```

in a Julia console. For more details, please refer to the [full documentation of MXNet.jl](http://mxnetjl.readthedocs.org/en/latest/user-guide/install.html).

## Scala Package Installation

For Linux/Mac users, we provide pre-built binary packages, with GPU or CPU-only supported.
You can use the following dependency in maven, change the artifactId according to your own architecture, e.g., `mxnet-full_2.10-osx-x86_64-cpu` for OSX (and cpu-only).

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-full_2.10-linux-x86_64-gpu</artifactId>
  <version>0.1.1</version>
</dependency>
```

In case your native environment is slightly different from which the assembly package provides, e.g., you use `openblas` instead of `atlas`, a more recommended way is to use `mxnet-core` and put the compiled Java native library somewhere in your load path.

```HTML
<dependency>
  <groupId>ml.dmlc.mxnet</groupId>
  <artifactId>mxnet-core_2.10</artifactId>
  <version>0.1.1</version>
</dependency>
```

To build with your own environment. First finish the [Build MXNet Library](#build-mxnet-library) step.
Then run following command from the root directory.

```bash
make scalapkg
```

Now you will find jars for `assembly`, `core` and `example` modules.
Also it produces the native library in `native/{your-architecture}/target`, which you can use to cooperate with the `core` module.

To install the scala package into your local maven repository, run

```bash
make scalainstall
```

## Docker Images

Builds of MXNet are available as [Docker](https://www.docker.com) images:
[MXNet Docker (CPU)](https://hub.docker.com/r/kaixhin/mxnet/) or
[MXNet Docker (CUDA)](https://hub.docker.com/r/kaixhin/cuda-mxnet/).
These are updated on a weekly basis with the latest builds of MXNet.
Examples of running bash in a Docker container are as follows:

```bash
sudo docker run -it kaixhin/mxnet
sudo nvidia-docker run -it kaixhin/cuda-mxnet:7.0
```

For a guide to Docker, see the [official docs](https://docs.docker.com).
CUDA support requires [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker).
For more details on how to use the MXNet Docker images,
consult the [source project](https://github.com/Kaixhin/dockerfiles).

## Build Dependent Libraries from Source

This section we provide instructions to build MXNet' dependent libraries from source. It is often useful in two situations:

- You use a low version or server linux, there is no according packages or the package versions are low by using `yum` or `apt-get`
- You do not have the root permission to install packages. In this case, you need to change the install directory from `/usr/local` into another one such as `${HOME}` in the following examples.

### Build GCC from Source

Building gcc needs 32-bit libc, you can install it by

- Ubuntu:  `sudo apt-get install libc6-dev-i386`
- Red Hat `sudo yum install glibc-devel.i686`
- CentOS 5.8, `sudo yum install glibc-devel.i386`
- CentOS 6 / 7, `sudo yum install glibc-devel.i686`

First download 
```bash
wget http://mirrors.concertpass.com/gcc/releases/gcc-4.8.5/gcc-4.8.5.tar.gz
tar -zxf gcc-4.8.5.tar.gz
cd gcc-4.8.5
./contrib/download_prerequisites
```

Then build
```
mkdir release && cd release
../configure --prefix=/usr/local --enable-languages=c,c++
make -j10
sudo make install
```

Finally you may want to add lib path in your `~/.bashrc`
```bash
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/lib64
```

### Build Opencv from Source

First download opencv 
```bash
wget https://codeload.github.com/opencv/opencv/zip/2.4.13
unzip 2.4.13
cd opencv-2.4.13
mkdir release
cd release/
```
Building opencv needs cmake, if you do not have cmake or your cmake verion is too low (e.g the one installed by default on RHEL), then 
```bash
wget https://cmake.org/files/v3.6/cmake-3.6.1-Linux-x86_64.tar.gz
tar -zxvf cmake-3.6.1-Linux-x86_64.tar.gz
alias cmake="cmake-3.6.1-Linux-x86_64/bin/cmake"
```
Now build opencv. We disable GPU support, which may significantly slow down to run a MXNet program on GPU. We also disable 1394 which may generate warning. 
```bash
cmake -D BUILD_opencv_gpu=OFF -D WITH_CUDA=OFF -D WITH_1394=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
make -j8
sudo make install
```
Finally, you may want to add the following into the end of your `~/.bashrc`:
```bash
export PKG_CONFIG_PATH=$PKG_CONFIG_PATH:/usr/local/lib/pkgconfig/
```