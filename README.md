# mumax5
GPU accelerated micromagnetic simulator.

# Dependencies

* MSVC 2019 (Windows)
* CUDA 10.x
* cmake>=3.18
* CPython 3
* pybind11
* NumPy
* matplotlib

# Installation from Source (Linux)

# Installation from Source (Windows)

1. Install Visual Studio 2019 and the desktop development with C++ workload
2. Install CUDA Toolkit 10.x
3. Install cmake
4. Download the pybind11 submodule with git
```
git submodule init
git submodule update
```
5. Install Python packages using conda
```
conda env create -f environment.yml
```
6. Build `mumax5` using `setuptools`
```
activate mumax5
python setup.py develop
```
or `conda`
```
conda activate mumax5
conda develop -b .
```

# Contributing
Contributions are gratefully accepted. To contribute code, fork our repo on github and send a pull request.
