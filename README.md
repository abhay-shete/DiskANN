# DiskANN

This is a local copy of the original [DiskANN](https://github.com/microsoft/DiskANN) project. Code has been modified so that it compiles on Mac.
Python bindings have also been added to the project. Pull Request to merge this code with the
original project will be raised soon.

The goal of the project is to build scalable, performant and cost-effective approximate nearest neighbor search algorithms.
This release has the code from the [DiskANN paper](https://papers.nips.cc/paper/9527-rand-nsg-fast-accurate-billion-point-nearest-neighbor-search-on-a-single-node.pdf) published in NeurIPS 2019, and improvements. 
This code reuses and builds upon some of the [code for NSG](https://github.com/ZJULearning/nsg) algorithm.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

See [guidelines](CONTRIBUTING.md) for contributing to this project.

## Mac Build:

Install the following packages

```bash
brew install gperftools boost llvm cmake clang-format
```

### Install MKL
Download and install [MKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html?operatingsystem=mac&distributions=webdownload&options=online)

### Build
```bash
mkdir build && cd build && cmake .. && make -j
```

### Python Bindings
Python bindings have been implemented under lib/pylib folder. 
setup.py file has been provided in the same directory. Module can be installed by executing the following commands

```bash
cd lib/pyblib
python setup.py install
```

#### Building Disk Index
```python
from diskann import IndexBuildParams, IndexSearchParams, DiskANN

# After running make, shared library used by Python will be 
# under build/lib/pylib directory
shared_lib_path = "./build/lib/pylib/libpydisk_index.dylib"
index_path = "./DiskANN_data/sift"

# np_vecs represents a numpy array containing input vectors
# If the code is run from the base directory, make sure that
# PYTHONPATH is set to ./lib/pylib e.g:
# export PYTHONPATH=$PYTHONPATH:./lib/pyblib

# IndexBuildParams and IndexSearchParams have been documented
# in lib/pylib/diskann.py
diskAnn = DiskANN(shared_lib_path)
idx_bld_params = IndexBuildParams(metric="l2", 
                                  graph_degree=32, 
                                  search_list_size=50,
                                  max_mem_build=1.0)

diskAnn.build_disk_index(index_path, np_vecs, idx_bld_params)

```

#### Searching Disk Index
```python

idx_srch_params = IndexSearchParams(num_nodes_to_cache=100000, 
                                    num_threads=32, 
                                    beam_width=4,
                                    search_list_size=60)
num_neighbours = 10
# query_res is an array (id, dist) tuples
query_res = diskAnn.search_disk_index(np_vecs, num_neighbours, idx_srch_params)

```
Explanatory notebook("Using DiskANN Python Bindings.ipynb") which downloads sample data and calls the DiskANN Python API to build and search index can be found in lib/pylib/

It took 15 minutes to build the index with the parameters documented in the API above for the SIFT 1 million dataset.
First 10k vectors from SIFT 1 million dataset were used as query vectors with num_neighbours=10.
Only 1 out of 10,000 queries failed in finding the correct vector in the top 10 nearset neighbours.

To calculate recall over num_neighbours i.e intersection set between search
results and the ground truth for top num_neighbours, following steps were carried out

1. ./build/tests/utils/compute_ground_truth was used to create ground truth data
2. ./build/tests/search_disk_index was used to search the index created from the Python API documented above which also provided Top-10 recall statistics. Following results were obtained with 10k query vectors from the SIFT 1 million dataset.
Please refer to the SSD_INDEX.md file under workflows/ for details about parameters for search_disk_index


Output of search_disk_index is as follows:

|     Search List Size   | Beamwidth |         QPS    | Mean Latency    |   99.9 Latency |      Mean IOs  |      CPU (s)   |    Recall@10 |
| --------------------   | --------- |  ------------- |  -------------- |   ------------ |   ------------ |   ------------ |   ---------- |
|        20              |        4  |       3708.90  |       7216.93   |     54430.00   |         8.27   |      4840.86   |        97.94 |
|        30              |        4  |       3145.50  |       8677.79   |     56792.00   |        11.69   |      5804.54   |        98.39 |
|        40              |        4  |       2716.36  |      10328.35   |     59509.00   |        15.24   |      7087.09   |        98.69 |


With higher values of graph degree(120) and search_list_size(150), index creation took about 80 minutes(5X as compared to original parameters). But the top-10 recall was more than 99% 




## Linux build:

Install the following packages through apt-get

```bash
sudo apt install cmake g++ libaio-dev libgoogle-perftools-dev clang-format libboost-dev
```

### Install Intel MKL
#### Ubuntu 20.04
```bash
sudo apt install libmkl-full-dev
```

#### Earlier versions of Ubuntu
Install Intel MKL either by downloading the [oneAPI MKL installer](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) or using [apt](https://software.intel.com/en-us/articles/installing-intel-free-libs-and-python-apt-repo) (we tested with build 2019.4-070 and 2022.1.2.146).

```
# OneAPI MKL Installer
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18487/l_BaseKit_p_2022.1.2.146.sh
sudo sh l_BaseKit_p_2022.1.2.146.sh -a --components intel.oneapi.lin.mkl.devel --action install --eula accept -s
```

### Build
```bash
mkdir build && cd build && cmake .. && make -j 
```

## Windows build:

The Windows version has been tested with the Enterprise editions of Visual Studio 2017 and Visual Studio 2019. It should work with the Community and Professional editions as well without any changes. 

**Prerequisites:**

* Install CMAKE (v3.15.2 or later) from https://cmake.org
* Install MKL from https://software.intel.com/en-us/mkl
* Install/download Boost from https://www.boost.org

* Environment variables: 
    * Set a new System environment variable, called INTEL_ROOT to the "windows" folder under your MKL installation
	   (For instance, if your install folder is "C:\Program Files (x86)\IntelSWtools", set INTEL_ROOT to "C:\Program Files (x86)\IntelSWtools\compilers_and_libraries\windows")
    * Set environment variable BOOST_ROOT to your boost folder.

**Build steps:**
-	Open a new command prompt window
-	Create a "build" directory under diskann
-	Change to the "build" directory and run  
```
<full-path-to-cmake>\cmake -G "Visual Studio 16 2019" -B. -A x64 ..
```
OR 
```
<full-path-to-cmake>\cmake -G "Visual Studio 15 2017" -B. -A x64 ..
```

**Note: Since VS comes with its own (older) version of cmake, you have to specify the full path to cmake to ensure that the right version is used.**
-	This will create a “diskann” solution file in the "build" directory
-	Open the "diskann" solution and build the “diskann” project. 
- 	Then build all the other binaries using the ALL_BUILD project that is part of the solution
- 	Generated binaries are stored in the diskann/x64/Debug or diskann/x64/Release directories.

To build from command line, change to the "build" directory and use msbuild to first build the "diskpriority_io" and "diskann_dll" projects. And then build the entire solution, as shown below.
```
msbuild src\dll\diskann.vcxproj
msbuild diskann.sln
```
Check msbuild docs for additional options including choosing between debug and release builds.


## Usage:

Please see the following pages on using the compiled code:

- [Commandline interface for building and search SSD based indices](workflows/SSD_index.md)  
- [Commandline interface for building and search in memory indices](workflows/in_memory_index.md) 
- To be added: Python interfaces and docker files
