// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the License);
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

// http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an AS IS BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
# C++ tests for PennyLane-Lightning

Gate implementations (kernels) are tested in `Test_GateImplementations_*.cpp` files.

As some of the kernels (AVX2 and AVX512) are only runnable on the corresponding architectures, we cannot guarantee testing all kernels on all CPU variants.
Even though it is possible to test available kernels by detecting the architecture, we currently use the approach below to simplify the test codes:

In `Test_GateImplementations_(Param|Nonparam|Matrix).cpp` files we test only the default kernels (`LM` and `PI`), of which both paradigms are architecture agnostic.

In `Test_GateImplementations_(Inverse|CompareKernels|Generator).cpp` files run tests registered to `DynamicDispatcher`. As we register kernels to `DynamicDispatcher` by detecting the runtime architecture, these files test all accessible kernels on the executing CPU architecture.
