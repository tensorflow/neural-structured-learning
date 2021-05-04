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

#include <glog/logging.h>

namespace carls {
namespace memory_store {

float DistanceUpperBound(MemoryDistanceConfig::DistanceType distance_type) {
  switch (distance_type) {
    case MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN:
      return -0.01f;  // exp(-inf).
    case MemoryDistanceConfig::SQUARED_L2:
      return std::numeric_limits<float>::max();
    default:
      LOG(FATAL) << "Unknown distance type: " << distance_type;
  }
}

float DistanceLowerBound(MemoryDistanceConfig::DistanceType distance_type) {
  switch (distance_type) {
    case MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN:
      return 1.0f;
    case MemoryDistanceConfig::SQUARED_L2:
      return 0;
    default:
      LOG(FATAL) << "Unknown distance type: " << distance_type;
  }
}

bool IsFurther(MemoryDistanceConfig::DistanceType distance_type,
               const float lhs, const float rhs) {
  switch (distance_type) {
    case MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN:
      return lhs < rhs;
    case MemoryDistanceConfig::SQUARED_L2:
      return lhs > rhs;
    default:
      LOG(FATAL) << "Unknown distance type: " << distance_type;
  }
}

}  // namespace memory_store
}  // namespace carls
