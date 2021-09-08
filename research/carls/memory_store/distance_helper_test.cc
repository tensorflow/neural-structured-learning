/*Copyright 2021 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "research/carls/memory_store/distance_helper.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace carls {
namespace memory_store {

TEST(DistanceHelperTest, DistanceUpperBound) {
  EXPECT_FLOAT_EQ(
      -0.01f, DistanceUpperBound(MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN));
  EXPECT_FLOAT_EQ(std::numeric_limits<float>::max(),
                  DistanceUpperBound(MemoryDistanceConfig::SQUARED_L2));
}

TEST(DistanceHelperTest, DistanceLowerBound) {
  EXPECT_FLOAT_EQ(
      1.0, DistanceLowerBound(MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN));
  EXPECT_FLOAT_EQ(0, DistanceLowerBound(MemoryDistanceConfig::SQUARED_L2));
}

TEST(DistanceHelperTest, IsFurther) {
  EXPECT_TRUE(IsFurther(MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN, 0.1, 0.8));
  EXPECT_TRUE(IsFurther(MemoryDistanceConfig::SQUARED_L2, 100, 20));
}

}  // namespace memory_store
}  // namespace carls
