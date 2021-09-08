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

#ifndef NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_MEMORY_STORE_DISTANCE_HELPER_H_
#define NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_MEMORY_STORE_DISTANCE_HELPER_H_

#include "research/carls/memory_store/memory_distance_config.pb.h"  // proto to pb

namespace carls {
namespace memory_store {

// Returns the furthest distance between two points based on given distance
// type.
float DistanceUpperBound(MemoryDistanceConfig::DistanceType distance_type);

// Returns the nearest distance between to points based on given distance type.
float DistanceLowerBound(MemoryDistanceConfig::DistanceType distance_type);

// Returns true if lhs is further away than rhs and false otherwise.
// The definition of 'further' depends on the distance type specified in
// MemoryStoreConfig.
bool IsFurther(MemoryDistanceConfig::DistanceType distance_type,
               const float lhs, const float rhs);

}  // namespace memory_store
}  // namespace carls

#endif  // NEURAL_STRUCTURED_LEARNING_RESEARCH_CARLS_MEMORY_STORE_DISTANCE_HELPER_H_
