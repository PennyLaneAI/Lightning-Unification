// Copyright 2018-2023 Xanadu Quantum Technologies Inc.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @file
 * Defines helper methods for PennyLane Lightning.
 */
#pragma once
#include <catch2/catch.hpp>

namespace Pennylane {

#define PL_REQUIRE_THROWS_MATCHES(expr, type, message_match)                   \
    REQUIRE_THROWS_AS(expr, type);                                             \
    REQUIRE_THROWS_WITH(expr, Catch::Matchers::Contains(message_match));
#define PL_CHECK_THROWS_MATCHES(expr, type, message_match)                     \
    CHECK_THROWS_AS(expr, type);                                               \
    CHECK_THROWS_WITH(expr, Catch::Matchers::Contains(message_match));

} // namespace Pennylane