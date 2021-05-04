/* Copyright 2021 Google LLC. All Rights Reserved.

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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/memory_store/gaussian_memory_config.pb.h"  // proto to pb
#include "research/carls/memory_store/memory_distance_config.pb.h"  // proto to pb
#include "research/carls/memory_store/memory_store.h"
#include "research/carls/testing/test_helper.h"

namespace carls {
namespace memory_store {

using ::testing::ElementsAre;
using ::testing::TempDir;

class GaussianMemoryTest : public ::testing::Test {
 protected:
  GaussianMemoryTest() {}

  std::unique_ptr<MemoryStore> CreateGaussianMemoryStore(
      const int per_cluster_buffer_size,
      const float distance_to_cluster_threshold, const int max_num_clusters,
      const int bootstrap_steps, const float min_variance,
      const MemoryDistanceConfig::DistanceType distance_type) {
    MemoryStoreConfig config;
    GaussianMemoryConfig gm_config;
    gm_config.set_per_cluster_buffer_size(per_cluster_buffer_size);
    gm_config.set_distance_to_cluster_threshold(distance_to_cluster_threshold);
    gm_config.set_max_num_clusters(max_num_clusters);
    gm_config.set_bootstrap_steps(bootstrap_steps);
    gm_config.set_min_variance(min_variance);
    gm_config.set_distance_type(distance_type);
    config.mutable_extension()->PackFrom(gm_config);
    return MemoryStoreFactory::Make(config);
  }
};

TEST_F(GaussianMemoryTest, Create) {
  // Invalid per_cluster_buffer_size.
  EXPECT_TRUE(
      CreateGaussianMemoryStore(
          /*per_cluster_buffer_size=*/0,
          /*distance_to_cluster_threshold=*/0.5,
          /*max_num_clusters=*/1,
          /*bootstrap_steps=*/1,
          /*min_variance=*/1,
          /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN) ==
      nullptr);
  // Invalid distance_to_cluster_threshold.
  EXPECT_TRUE(
      CreateGaussianMemoryStore(
          /*per_cluster_buffer_size=*/1,
          /*distance_to_cluster_threshold=*/-1,
          /*max_num_clusters=*/1,
          /*bootstrap_steps=*/1,
          /*min_variance=*/1,
          /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN) ==
      nullptr);
  // Invalid max_num_clusters.
  EXPECT_TRUE(
      CreateGaussianMemoryStore(
          /*per_cluster_buffer_size=*/1,
          /*distance_to_cluster_threshold=*/0.5,
          /*max_num_clusters=*/0,
          /*bootstrap_steps=*/1,
          /*min_variance=*/1,
          /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN) ==
      nullptr);
  // Invalid distance_type.
  EXPECT_TRUE(CreateGaussianMemoryStore(
                  /*per_cluster_buffer_size=*/1,
                  /*distance_to_cluster_threshold=*/0.5,
                  /*max_num_clusters=*/1,
                  /*bootstrap_steps=*/1,
                  /*min_variance=*/1,
                  /*distance_type=*/MemoryDistanceConfig::DEFAULT_UNKNOWN) ==
              nullptr);
  // Invalid min_variance.
  EXPECT_TRUE(
      CreateGaussianMemoryStore(
          /*per_cluster_buffer_size=*/1,
          /*distance_to_cluster_threshold=*/0.5,
          /*max_num_clusters=*/1,
          /*bootstrap_steps=*/1,
          /*min_variance=*/0,
          /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN) ==
      nullptr);
  EXPECT_TRUE(
      CreateGaussianMemoryStore(
          /*per_cluster_buffer_size=*/1,
          /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
          /*bootstrap_steps=*/1,
          /*min_variance=*/1,
          /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN) !=
      nullptr);
}

TEST_F(GaussianMemoryTest, BatchLookup_CwiseMeanGaussianDistance) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/1,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/1,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 2
  )pb");
  std::vector<MemoryLookupResult> results;

  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 1 value: 2 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));

  // Distance exp(-dist^2/2) where dist = ((2 - 1)^2 + (3 - 2)^2) / 2 = 1
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 2
    value: 3
  )pb");
  ASSERT_OK(memory_store->BatchLookup(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 1 value: 2 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 0.60653067
                cluster_index: 0
              )pb")));
}

TEST_F(GaussianMemoryTest, BatchLookup_SquareToCenterDistance) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/1,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/1,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::SQUARED_L2);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 2
  )pb");
  std::vector<MemoryLookupResult> results;

  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 1 value: 2 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 0
                cluster_index: 0
              )pb")));

  // Distance (2 - 1)^2 + (3 - 2)^2) = 2
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 2
    value: 3
  )pb");
  ASSERT_OK(memory_store->BatchLookup(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 1 value: 2 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 2
                cluster_index: 0
              )pb")));
}

TEST_F(GaussianMemoryTest, BatchLookup_SingleInput) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/1,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/1,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  // Empty cluster for inference mode.
  EXPECT_NOT_OK(memory_store->BatchLookup(inputs, &results));

  // First lookup with update, returns single cluster.
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));

  // Another input [1, 0] with distance:
  // exp(-1/(2*1.001^2)) = 0.7807309
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookup(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 0.7788008
                cluster_index: 0
              )pb")));

  // A third input [100, 0] that is too far away so the distance becomes 0.
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookup(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 0
                cluster_index: 0
              )pb")));
}

TEST_F(GaussianMemoryTest, BatchLookup_BatchInput) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/1,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/1,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);
  std::vector<MemoryLookupResult> results;

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  // Adds a new cluster.
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");

  // Batch lookup, the results should be the same as single lookup above.
  ASSERT_OK(memory_store->BatchLookup(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 0 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 1
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 0 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 0.7788008
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 0 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 0
                                     cluster_index: 0
                                   )pb")));
}

TEST_F(GaussianMemoryTest, BatchLookupWithUpdate_SingleInput_SingleBufferSize) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/1,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  // First lookup, returns single cluster.
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));

  // When another input [1, 0] is added, the old center will be preempted.
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 1 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));

  // A third input [100, 0] is add, the old center will be preempted.
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 100 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));
}

TEST_F(GaussianMemoryTest, BatchLookupWithUpdate_SingleInput_MultiBufferSize) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  // First lookup, returns single cluster.
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));

  // Another input [1, 0] that is close to [0, 0].
  // New mean: ([1, 0] + [0, 0]) / 2 = [0.5, 0].
  // New variance: [(1 - 0.5)^2 + (0 - 0.5)^2, 0] / 2 = [0.25, 0]
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0.5 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 0.9394131
                cluster_index: 0
              )pb")));

  // A third input [100, 0] that is far away to [0.5, 0].
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 33.666668 value: 0 }
                  variance { value: 2200.222 value: 1 }
                }
                distance_to_cluster: 0.6065536
                cluster_index: 0
              )pb")));
}

// This is an  uncommon case that the buffer size is smaller than the batch
// size, in which the Gaussian center becomes that of the last element in the
// batch.
TEST_F(GaussianMemoryTest, BatchLookupWithUpdate_MultiInput_SingleBufferSize) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/1,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 100 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 0
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 100 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 0
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 100 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 1
                                     cluster_index: 0
                                   )pb")));
}

TEST_F(GaussianMemoryTest, BatchLookupWithUpdate_MultiInput_MultiBufferSize) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 33.666668 value: 0 }
                                       variance { value: 2200.222 value: 1 }
                                     }
                                     distance_to_cluster: 0.87916076
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 33.666668 value: 0 }
                                       variance { value: 2200.222 value: 1 }
                                     }
                                     distance_to_cluster: 0.8858121
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 33.666668 value: 0 }
                                       variance { value: 2200.222 value: 1 }
                                     }
                                     distance_to_cluster: 0.6065536
                                     cluster_index: 0
                                   )pb")));
}

// For buffer_size = 1,BatchLookupWithGrow == BatchLookupWithUpdate.
TEST_F(GaussianMemoryTest, BatchLookupWithGrow_SingleInput_SingleBufferSize) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/1,
      /*distance_to_cluster_threshold=*/0.5, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  // First lookup, returns single cluster.
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));

  // When another input [1, 0] is added, the old center will be preempted.
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookupWithGrow(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 1 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));

  // A third input [100, 0] is add, the old center will be preempted.
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookupWithGrow(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 100 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));
}

TEST_F(GaussianMemoryTest, BatchLookupWithGrow_SingleInput_MultiCluster) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.7, /*max_num_clusters=*/3,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  // First lookup, returns single cluster.
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 0
              )pb")));

  // Another input [1, 0] that is close to [0, 0].
  // New mean: ([1, 0] + [0, 0]) / 2 = [0.5, 0].
  // New variance: [(1 - 0.5)^2 + (0 - 0.5)^2, 0] / 2 = [0.25, 0]
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookupWithGrow(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 0.5 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 0.9394131
                cluster_index: 0
              )pb")));

  // A third input [100, 0] that is far away to [0.5, 0], a new cluster is
  // formed.
  *inputs.Mutable(0) = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  ASSERT_OK(memory_store->BatchLookupWithGrow(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 100 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 1
              )pb")));
}

// For single cluster, BatchLookupWithGrow == BatchLookupWithUpdate.
TEST_F(GaussianMemoryTest, BatchLookupWithGrow_MultiInput_SingleCluster) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.7, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  ASSERT_OK(memory_store->BatchLookupWithGrow(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 33.666668 value: 0 }
                                       variance { value: 2200.222 value: 1 }
                                     }
                                     distance_to_cluster: 0.87916076
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 33.666668 value: 0 }
                                       variance { value: 2200.222 value: 1 }
                                     }
                                     distance_to_cluster: 0.8858121
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 33.666668 value: 0 }
                                       variance { value: 2200.222 value: 1 }
                                     }
                                     distance_to_cluster: 0.6065536
                                     cluster_index: 0
                                   )pb")));
}

// For single cluster, BatchLookupWithGrow == BatchLookupWithUpdate.
TEST_F(GaussianMemoryTest, BatchLookupWithGrow_MultiInput_MultiCluster) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.7, /*max_num_clusters=*/3,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  // Increases the update_steps_counter_ by 1.
  EXPECT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");

  // The point [0, 0] is added into the first cluster twice, so its mean becomes
  // (0 + 0 + 1) / 3.
  // A new cluster is created for [100, 0].
  ASSERT_OK(memory_store->BatchLookupWithGrow(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 0.33333334 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 0.97260445
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 0.33333334 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 0.89483935
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 100 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 1
                                     cluster_index: 1
                                   )pb")));
}

TEST_F(GaussianMemoryTest, Export) {
  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.7, /*max_num_clusters=*/3,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 0
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;

  // Increases the update_steps_counter_ by 1.
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 1
    value: 0
  )pb");
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");

  // The point [0, 0] is added into the first cluster twice, so its
  // mean becomes (0 + 0 + 1) / 3.
  // A new cluster is created for [100, 0].
  ASSERT_OK(memory_store->BatchLookupWithGrow(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 0.33333334 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 0.97260445
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 0.33333334 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 0.89483935
                                     cluster_index: 0
                                   )pb"),
                                   EqualsProto<MemoryLookupResult>(R"pb(
                                     gaussian_cluster {
                                       mean { value: 100 value: 0 }
                                       variance { value: 1 value: 1 }
                                     }
                                     distance_to_cluster: 1
                                     cluster_index: 1
                                   )pb")));

  std::string checkpoint_path;
  ASSERT_OK(memory_store->Export(TempDir(), "name", &checkpoint_path));

  GaussianMemoryCheckpointMetaData meta_data;
  ASSERT_OK(ReadTextProto(checkpoint_path, &meta_data));
  EXPECT_THAT(meta_data, EqualsProto<GaussianMemoryCheckpointMetaData>(R"pb(
                cluster_data {
                  gaussian_cluster {
                    mean { value: 0.33333334 value: 0 }
                    variance { value: 1 value: 1 }
                  }
                  activation_instance { value: 0 value: 0 }
                  activation_instance { value: 0 value: 0 }
                  activation_instance { value: 1 value: 0 }
                }
                cluster_data {
                  gaussian_cluster {
                    mean { value: 100 value: 0 }
                    variance { value: 1 value: 1 }
                  }
                  activation_instance { value: 100 value: 0 }
                }
              )pb"));
}

TEST_F(GaussianMemoryTest, Import) {
  // Save a checkpoint.
  auto meta_data = ParseTextProtoOrDie<GaussianMemoryCheckpointMetaData>(R"pb(
    cluster_data {
      gaussian_cluster {
        mean { value: 0.33333334 value: 0 }
        variance { value: 1 value: 1 }
      }
      activation_instance { value: 0 value: 0 }
      activation_instance { value: 0 value: 0 }
      activation_instance { value: 1 value: 0 }
    }
    cluster_data {
      gaussian_cluster {
        mean { value: 100 value: 0 }
        variance { value: 1 value: 1 }
      }
      activation_instance { value: 100 value: 0 }
    }
  )pb");
  const std::string ckpt_path = JoinPath(TempDir(), "mem1");
  ASSERT_OK(WriteTextProto(ckpt_path, meta_data, /*can_overwrite=*/true));

  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.7, /*max_num_clusters=*/3,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  // Looks up the center of cluster 1 and it should return the correct cluster
  // id and distance.
  ASSERT_OK(memory_store->Import(ckpt_path));
  google::protobuf::RepeatedPtrField<EmbeddingVectorProto> inputs;/*proto2*/
  *inputs.Add() = ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
    value: 100
    value: 0
  )pb");
  std::vector<MemoryLookupResult> results;
  ASSERT_OK(memory_store->BatchLookupWithUpdate(inputs, &results));
  EXPECT_THAT(results, ElementsAre(EqualsProto<MemoryLookupResult>(R"pb(
                gaussian_cluster {
                  mean { value: 100 value: 0 }
                  variance { value: 1 value: 1 }
                }
                distance_to_cluster: 1
                cluster_index: 1
              )pb")));
}

TEST_F(GaussianMemoryTest, Import_FailedTooManyClusters) {
  // Save a checkpoint with too many clusters.
  auto meta_data = ParseTextProtoOrDie<GaussianMemoryCheckpointMetaData>(R"pb(
    cluster_data {
      gaussian_cluster {
        mean { value: 0.33333334 value: 0 }
        variance { value: 1 value: 1 }
      }
      activation_instance { value: 0 value: 0 }
    }
    cluster_data {
      gaussian_cluster {
        mean { value: 100 value: 0 }
        variance { value: 1 value: 1 }
      }
      activation_instance { value: 100 value: 0 }
    }
  )pb");
  const std::string ckpt_path = JoinPath(TempDir(), "mem1");
  ASSERT_OK(WriteTextProto(ckpt_path, meta_data, /*can_overwrite=*/true));

  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.7, /*max_num_clusters=*/1,
      /*bootstrap_steps=*/0,
      /*min_variance=*/1,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  // Looks up the center of cluster 1 and it should return the correct cluster
  // id and distance.
  EXPECT_ERROR_CONTAIN(memory_store->Import(ckpt_path), "Too many clusters.");
}

TEST_F(GaussianMemoryTest, Import_FailedTooSmallVariance) {
  // Save a checkpoint with too many clusters.
  auto meta_data = ParseTextProtoOrDie<GaussianMemoryCheckpointMetaData>(R"pb(
    cluster_data {
      gaussian_cluster {
        mean { value: 0.33333334 value: 0 }
        variance { value: 1 value: 4 }
      }
      activation_instance { value: 0 value: 0 }
    }
  )pb");
  const std::string ckpt_path = JoinPath(TempDir(), "mem1");
  ASSERT_OK(WriteTextProto(ckpt_path, meta_data, /*can_overwrite=*/true));

  auto memory_store = CreateGaussianMemoryStore(
      /*per_cluster_buffer_size=*/3,
      /*distance_to_cluster_threshold=*/0.7, /*max_num_clusters=*/2,
      /*bootstrap_steps=*/0,
      /*min_variance=*/2,
      /*distance_type=*/MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN);

  // Looks up the center of cluster 1 and it should return the correct cluster
  // id and distance.
  EXPECT_ERROR_CONTAIN(memory_store->Import(ckpt_path),
                       "Too small variance value.");
}

}  // namespace memory_store
}  // namespace carls
