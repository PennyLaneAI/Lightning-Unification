# PennyLane-Lightning Plugin

The PennyLane-Lightning plugin provides a fast state-vector simulator written in C++.

[PennyLane](https://docs.pennylane.ai) is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

## Features

Combine PennyLane-Lightning's high-performance simulators with PennyLane's
  automatic differentiation and optimization.

## Installation

PennyLane-Lightning requires Python version 3.8 and above. It can be installed using ``pip``:

```console
pip install pennylane-lightning
```

To build PennyLane-Lightning from source you can run

```console
pip install pybind11 pennylane-lightning --no-binary :all:
```

A C++ compiler such as ``g++``, ``clang++``, or ``MSVC`` is required.
On Debian-based systems, this can be installed via ``apt``:

```console
sudo apt install g++
```

On MacOS, we recommend using the latest version of ``clang++`` and ``libomp``:

```console
brew install llvm libomp
```

The [pybind11](https://pybind11.readthedocs.io/en/stable/>) library is also used for binding the
C++ functionality to Python.

Alternatively, for development and testing, you can install by cloning the repository:

```console
git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning
pip install -r requirements.txt
pip install -e .
```

Note that subsequent calls to ``pip install -e .`` will use cached binaries stored in the
``build`` folder. Run ``make clean`` if you would like to recompile.

You can also pass ``cmake`` options with ``CMAKE_ARGS`` as follows:

```console
CMAKE_ARGS="-DENABLE_OPENMP=OFF -DENABLE_BLAS=OFF -DENABLE_KOKKOS=OFF" pip install -e . -vv
```

or with ``build_ext`` and the ``--define`` flag as follows:

```console
python3 setup.py build_ext -i --define="ENABLE_OPENMP=OFF;ENABLE_BLAS=OFF;ENABLE_KOKKOS=OFF"
python3 setup.py develop
```

## GPU support

For GPU support, [PennyLane-Lightning-GPU](https://github.com/PennyLaneAI/pennylane-lightning-gpu)
can be installed by providing the optional ``[gpu]`` tag:

    $ pip install pennylane-lightning[gpu]

For more information, please refer to the PennyLane Lightning GPU [documentation](https://docs.pennylane.ai/projects/lightning-gpu).

## Testing

To test that the plugin is working correctly you can test the Python code within the cloned
repository:

```console
make test-python
```

while the C++ code can be tested with

```console
make test-cpp
```

## CMake Support

One can also build the plugin using CMake:

```console
cmake -S. -B build
cmake --build build
```

To test the C++ code:

```console
mkdir build && cd build
cmake -DBUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Debug ..
make
```

Other supported options are

- ``-DENABLE_WARNINGS=ON``
- ``-DENABLE_NATIVE=ON`` (for ``-march=native``)
- ``-DENABLE_BLAS=ON``
- ``-DENABLE_OPENMP=ON``
- ``-DENABLE_KOKKOS=ON``
- ``-DENABLE_CLANG_TIDY=ON``

## Compile on Windows with MSVC

You can also compile Pennylane-Lightning on Windows using
[Microsoft Visual C++](https://visualstudio.microsoft.com/vs/features/cplusplus/) compiler.
You need [cmake](https://cmake.org/download/) and appropriate Python environment
(e.g. using [Anaconda](https://www.anaconda.com/)).


We recommend to use ``[x64 (or x86)] Native Tools Command Prompt for VS [version]`` for compiling the library.
Be sure that ``cmake`` and ``python`` can be called within the prompt.

```console
cmake --version
python --version
```

Then a common command will work.

```console
pip install -r requirements.txt
pip install -e .
```

Note that OpenMP and BLAS are disabled in this setting.

Please refer to the [plugin documentation](https://docs.pennylane.ai/projects/lightning/) as well as to the [PennyLane documentation](https://docs.pennylane.ai/) for further reference.


## Docker Support
One can also build the Pennylane-Lightning image using Docker:

```console
git clone https://github.com/PennyLaneAI/pennylane-lightning.git
cd pennylane-lightning
docker build -t lightning/base -f docker/Dockerfile .
```

Please refer to the [PennyLane installation](https://docs.pennylane.ai/en/stable/development/guide/installation.html#installation) for detailed description about PennyLane Docker support.

## Contributing

We welcome contributions - simply fork the repository of this plugin, and then make a
[pull request](https://help.github.com/articles/about-pull-requests/) containing your contribution.
All contributors to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

## Authors

PennyLane-Lightning is the work of [many contributors](https://github.com/PennyLaneAI/pennylane-lightning/graphs/contributors).

If you are doing research using PennyLane and PennyLane-Lightning, please cite [our paper](https://arxiv.org/abs/1811.04968):

> Ville Bergholm et al. *PennyLane: Automatic differentiation of hybrid quantum-classical
> computations.* 2018. arXiv:1811.04968

## Support

- **Source Code:** https://github.com/PennyLaneAI/pennylane-lightning
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-lightning/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

## License

The PennyLane lightning plugin is **free** and **open source**, released under
the [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

## Acknowledgements

PennyLane Lightning makes use of the following libraries and tools, which are under their own respective licenses:

- **pybind11:** https://github.com/pybind/pybind11
- **Kokkos Core:** https://github.com/kokkos/kokkos
- **Kokkos Kernels:** https://github.com/kokkos/kokkos-kernels
