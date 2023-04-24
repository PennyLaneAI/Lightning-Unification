PYTHON3 := $(shell which python3 2>/dev/null)

PYTHON := python3
TESTRUNNER := -m pytest tests --tb=short

.PHONY: help
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  docs                             to generate documents"
	@echo "  clean                            to delete all temporary, cache, and build files"
	@echo "  clean-docs                       to delete all built documentation"
	@echo "  test                             to run the test suite"
	@echo "  test-cpp [backend=?] [verbose=?] to run the C++ test suite;"
	@echo "                                   use with 'backend=lightning_kokkos' for Kokkos device. Default: lightning_qubit"
	@echo "                                   use with 'verbose=1' for building with verbose flag"
	@echo "  test-python                      to run the Python test suite"
	@echo "  format [check=1]                 to apply C++ and Python formatter;"
	@echo "                                   use with 'check=1' to check instead of modify (requires black and clang-format)"
	@echo "  format [version=?]               to apply C++ and Python formatter;"
	@echo "                                   use with 'version={version}' to check or modify with clang-format-{version} instead of clang-format"

.PHONY : clean
clean:
	find . -type d -name '__pycache__' -exec rm -r {} \+
	rm -rf build
	rm -rf Build BuildTests BuildTidy BuildGBench
	rm -rf pennylane_lightning/lightning_qubit_ops*

build:
	rm -rf ./Build
	cmake -BBuild -DENABLE_BLAS=ON -DENABLE_KOKKOS=ON -DENABLE_WARNINGS=ON -DPL_BACKEND=$(if $(backend:-=),$(backend),lightning_qubit)
ifdef verbose
	cmake --build ./Build --verbose
else
	cmake --build ./Build
endif

test-cpp:
	rm -rf ./BuildTests
	cmake -BBuildTests -DBUILD_TESTS=ON -DENABLE_WARNINGS=ON -DPL_BACKEND=$(if $(backend:-=),$(backend),lightning_qubit)
ifdef verbose
	cmake --build ./BuildTests --verbose
else
	cmake --build ./BuildTests
endif
	cmake --build ./BuildTests --target test

test-cpp-blas:
	rm -rf ./BuildTests
	cmake -BBuildTests -DBUILD_TESTS=ON  -DENABLE_BLAS=ON -DENABLE_WARNINGS=ON -DPL_BACKEND=$(if $(backend:-=),$(backend),lightning_qubit)
ifdef verbose
	cmake --build ./BuildTests --verbose
else
	cmake --build ./BuildTests
endif
	cmake --build ./BuildTests --target test

.PHONY: format format-cpp
format: format-cpp format-python

format-cpp:
ifdef check
	./bin/format --check --cfversion $(if $(version:-=),$(version),0) ./pennylane_lightning
else
	./bin/format --cfversion $(if $(version:-=),$(version),0) ./pennylane_lightning
endif

format-python:
ifdef check
	black -l 100 ./pennylane_lightning/ ./tests --check
else
	black -l 100 ./pennylane_lightning/ ./tests
endif
