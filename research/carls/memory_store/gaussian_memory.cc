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

#include <cstdint>
#include <list>

#include "google/protobuf/any.pb.h"  // proto to pb
#include "absl/status/status.h"
#include "third_party/eigen3/Eigen/Core"
#include "research/carls/base/embedding_helper.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/base/status_helper.h"
#include "research/carls/memory_store/distance_helper.h"
#include "research/carls/memory_store/gaussian_memory_config.pb.h"  // proto to pb
#include "research/carls/memory_store/memory_store.h"

namespace carls {
namespace memory_store {
namespace {

using Eigen::VectorXf;

// In-memory representation of a Gaussian cluster.
struct InMemoryClusterData {
  VectorXf mean;
  VectorXf variance;
  std::list<VectorXf> instances;
};

constexpr char kDataOutput[] = "gaussian_memory_metadata.pbtext";

}  // namespace

// A GaussianMemory represents the input activations using Gaussian clusters,
// namely {(center, variance)}.
class GaussianMemory : public MemoryStore {
 public:
  explicit GaussianMemory(const MemoryStoreConfig& config);

 private:
  // Represents (cluster index, distance, is_first) where is_first is true if
  // the returned cluster is dynamically created as the first one.
  using DistanceToNearestCluster = std::tuple<int, float, bool>;

  // Internal implementation of BatchLookup.
  absl::Status BatchLookupInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) override;

  // Internal implementation of BatchLookupWithUpdate.
  absl::Status BatchLookupWithUpdateInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) override;

  // Internal implementation of BatchLookupWithGrow.
  absl::Status BatchLookupWithGrowInternal(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      std::vector<MemoryLookupResult>* results) override;

  // Internal implementation of the Export() method.
  absl::Status ExportInternal(const std::string& dir,
                              std::string* exported_path) override;

  // Internal implementation of the Import() method.
  absl::Status ImportInternal(const std::string& saved_path) override;

  // Converts given inputs and cluster indices into an array of
  // MemoryLookupResult.
  absl::Status ProcessResults(
      const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
      const std::vector<int>& cluster_indices,
      std::vector<MemoryLookupResult>* results);

  // Computes the distance between a given input and a Gaussian cluster based on
  // gm_config_.distance_type().
  float ComputeDistance(const EmbeddingVectorProto& input,
                        const InMemoryClusterData& cluster);

  // Finds the closest cluster to the given input. It creates a new cluster if
  // cluster_list_ is empty.
  DistanceToNearestCluster FindNearestCluster(
      const EmbeddingVectorProto& input);

  // Creates a new cluster based on given input.
  // Returns the index of the new cluster.
  int AddNewCluster(const EmbeddingVectorProto& input);

  // Adds input into the specified cluster and updates the cluster.
  void AddInputToCluster(const EmbeddingVectorProto& input,
                         const int cluster_index);

  // Converts from InMemoryClusterData to GaussianCluster.
  GaussianCluster ConvertToGaussianCluster(const InMemoryClusterData& cluster);

  // Converts internal InMemoryClusterData into GaussianMemoryCheckpointMetaData
  // for checkpoint output.
  GaussianMemoryCheckpointMetaData ConvertToCheckpointMetaData();

  // An instance of GaussianMemoryConfig in MemoryStoreConfig.
  const GaussianMemoryConfig gm_config_;

  // The furthest distance used for initializing a nearest cluster search.
  // For example, if the distance type is SQUARED_L2, the futherest distance
  // can be defined to be a very large value like 1e10.
  const float furthest_distance_;

  // Keeps track of the number of clusters.
  std::atomic<int> cluster_counter_;

  // Counts the steps ran for either update or grow.
  std::atomic<int64_t> update_steps_counter_;

  // List of clusters.
  absl::Mutex mu_;
  std::vector<InMemoryClusterData> cluster_list_ ABSL_GUARDED_BY(mu_);
};

REGISTER_MEMORY_STORE_FACTORY(
    GaussianMemoryConfig,
    [](const MemoryStoreConfig& config) -> std::unique_ptr<MemoryStore> {
      GaussianMemoryConfig gm_config;
      config.extension().UnpackTo(&gm_config);
      if (gm_config.per_cluster_buffer_size() <= 0) {
        LOG(ERROR) << "Invalid per_cluster_buffer_size: "
                   << gm_config.per_cluster_buffer_size();
        return nullptr;
      }
      if (gm_config.max_num_clusters() <= 0) {
        LOG(ERROR) << "Invalid max_num_clusters: "
                   << gm_config.max_num_clusters();
        return nullptr;
      }
      if (gm_config.distance_to_cluster_threshold() < 0) {
        LOG(ERROR) << "Invalid distance_to_cluster_threshold: "
                   << gm_config.distance_to_cluster_threshold();
        return nullptr;
      }
      if (gm_config.min_variance() <= 0) {
        LOG(ERROR) << "Invalid min_variance: " << gm_config.min_variance();
        return nullptr;
      }
      if (gm_config.distance_type() == MemoryDistanceConfig::DEFAULT_UNKNOWN) {
        LOG(ERROR) << "Unknown distance_type.";
        return nullptr;
      }

      return absl::make_unique<GaussianMemory>(config);
    });

GaussianMemory::GaussianMemory(const MemoryStoreConfig& config)
    : MemoryStore(config),
      gm_config_([&config]() {
        GaussianMemoryConfig gm_config;
        config.extension().UnpackTo(&gm_config);
        return gm_config;
      }()),
      furthest_distance_(DistanceUpperBound(gm_config_.distance_type())),
      cluster_counter_(0),
      update_steps_counter_(0) {}

absl::Status GaussianMemory::BatchLookupInternal(
    const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
    std::vector<MemoryLookupResult>* results) {
  RET_CHECK_TRUE(cluster_counter_ > 0);
  std::vector<int> cluster_indices;
  cluster_indices.reserve(inputs.size());
  for (const auto& input : inputs) {
    const auto nc_tuple = FindNearestCluster(input);
    const int index = std::get<0>(nc_tuple);
    cluster_indices.push_back(index);
  }

  return ProcessResults(inputs, cluster_indices, results);
}

absl::Status GaussianMemory::BatchLookupWithUpdateInternal(
    const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
    std::vector<MemoryLookupResult>* results) {
  std::vector<int> cluster_indices;
  cluster_indices.reserve(inputs.size());
  for (const auto& input : inputs) {
    const auto nc_tuple = FindNearestCluster(input);
    const int index = std::get<0>(nc_tuple);
    const bool is_first = std::get<2>(nc_tuple);
    if (!is_first) {
      // Updates the new mean and variance.
      // If is_first = true, the single data point is already added into the new
      // cluster.
      AddInputToCluster(input, index);
    }
    cluster_indices.push_back(index);
  }

  ++update_steps_counter_;
  return ProcessResults(inputs, cluster_indices, results);
}

absl::Status GaussianMemory::BatchLookupWithGrowInternal(
    const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
    std::vector<MemoryLookupResult>* results) {
  std::vector<int> cluster_indices;
  cluster_indices.reserve(inputs.size());
  for (const auto& input : inputs) {
    const auto nc_tuple = FindNearestCluster(input);
    const int index = std::get<0>(nc_tuple);
    const float distance = std::get<1>(nc_tuple);
    const bool is_first = std::get<2>(nc_tuple);
    if (is_first) {
      cluster_indices.push_back(index);
      continue;
    }
    const bool can_add_new_cluster =
        (cluster_counter_ < gm_config_.max_num_clusters()) &&
        (update_steps_counter_ > gm_config_.bootstrap_steps());
    if (can_add_new_cluster &&
        IsFurther(gm_config_.distance_type(), distance,
                  gm_config_.distance_to_cluster_threshold())) {
      // Too far away from existing clusters, add a new cluster instead.
      cluster_indices.push_back(AddNewCluster(input));
      continue;
    }
    // update the new mean and variance.
    AddInputToCluster(input, index);
    cluster_indices.push_back(index);
  }

  ++update_steps_counter_;
  return ProcessResults(inputs, cluster_indices, results);
}

absl::Status GaussianMemory::ExportInternal(const std::string& dirname,
                                            std::string* exported_path) {
  *exported_path = JoinPath(dirname, kDataOutput);
  RET_CHECK_OK(WriteTextProto(*exported_path, ConvertToCheckpointMetaData(),
                              /*can_overwrite=*/true));
  return absl::OkStatus();
}

absl::Status GaussianMemory::ProcessResults(
    const google::protobuf::RepeatedPtrField<EmbeddingVectorProto>& inputs,/*proto2*/
    const std::vector<int>& cluster_indices,
    std::vector<MemoryLookupResult>* results) {
  RET_CHECK_TRUE(inputs.size() == cluster_indices.size());
  results->clear();
  results->reserve(cluster_indices.size());
  absl::MutexLock l(&mu_);
  for (size_t i = 0; i < cluster_indices.size(); ++i) {
    const int cluster_index = cluster_indices[i];
    MemoryLookupResult result;
    *result.mutable_gaussian_cluster() =
        ConvertToGaussianCluster(cluster_list_[cluster_index]);
    result.set_distance_to_cluster(
        ComputeDistance(inputs[i], cluster_list_[cluster_index]));
    result.set_cluster_index(cluster_index);
    results->push_back(std::move(result));
  }
  return absl::OkStatus();
}

float GaussianMemory::ComputeDistance(const EmbeddingVectorProto& input,
                                      const InMemoryClusterData& cluster) {
  VectorXf input_vec = std::move(ToInMemoryEmbeddingVector(input).vec);
  if (gm_config_.distance_type() == MemoryDistanceConfig::CWISE_MEAN_GAUSSIAN) {
    // Computes the probability based on the following formula:
    // dist(x, mean) = exp(-1/2*(x-mean)^T\Sigma^{-1}(x - mean))
    VectorXf dist = input_vec - cluster.mean;
    for (int i = 0; i < dist.size(); ++i) {
      dist(i) = dist(i) * dist(i) / cluster.variance(i);
    }
    return std::exp(-dist.mean() / 2);
  }
  if (gm_config_.distance_type() == MemoryDistanceConfig::SQUARED_L2) {
    // Compute the L2 distance between input and the mean vector.
    VectorXf dist = input_vec - cluster.mean;
    return dist.squaredNorm();
  }
  LOG(FATAL) << "Unknown distance type: " << gm_config_.distance_type();
}

GaussianMemory::DistanceToNearestCluster GaussianMemory::FindNearestCluster(
    const EmbeddingVectorProto& input) {
  if (cluster_counter_ == 0) {
    AddNewCluster(input);
    return std::make_tuple(/*index*/ 0,
                           DistanceLowerBound(gm_config_.distance_type()),
                           /*is_first*/ true);
  }

  absl::MutexLock l(&mu_);
  int closest_cluster_index = -1;
  float closest_distance = furthest_distance_;
  for (int i = 0; i < cluster_list_.size(); ++i) {
    float dist = ComputeDistance(input, cluster_list_[i]);
    if (IsFurther(gm_config_.distance_type(), closest_distance, dist)) {
      closest_distance = dist;
      closest_cluster_index = i;
    }
  }
  QCHECK_GE(closest_cluster_index, 0) << "Possible nan values encountered.";
  return std::make_tuple(closest_cluster_index, closest_distance,
                         /*is_first*/ false);
}

int GaussianMemory::AddNewCluster(const EmbeddingVectorProto& input) {
  auto input_vec = std::move(ToInMemoryEmbeddingVector(input).vec);
  absl::MutexLock l(&mu_);
  InMemoryClusterData cluster_data;
  cluster_data.mean = input_vec;
  cluster_data.variance =
      VectorXf::Ones(input_vec.size()) * gm_config_.min_variance();
  cluster_data.instances.push_back(std::move(input_vec));
  cluster_list_.push_back(std::move(cluster_data));
  ++cluster_counter_;
  return cluster_list_.size() - 1;
}

void GaussianMemory::AddInputToCluster(const EmbeddingVectorProto& input,
                                       const int cluster_index) {
  VectorXf input_vec = std::move(ToInMemoryEmbeddingVector(input).vec);

  absl::MutexLock l(&mu_);
  CHECK_LT(cluster_index, cluster_list_.size());
  auto* cluster = &cluster_list_[cluster_index];
  VectorXf total_mean = cluster->mean * cluster->instances.size() + input_vec;
  // Checks that per_cluster_buffer_size is not exceeded.
  if (cluster->instances.size() >= gm_config_.per_cluster_buffer_size()) {
    total_mean -= cluster->instances.front();
    cluster->instances.pop_front();
  }
  cluster->instances.push_back(input_vec);
  cluster->mean = total_mean / cluster->instances.size();

  // Computes element-wise variance.
  VectorXf total_variance;
  if (cluster->instances.size() == 1) {
    cluster->variance =
        VectorXf::Ones(total_mean.size()) * gm_config_.min_variance();
    return;
  }
  total_variance = VectorXf::Zero(total_mean.size());
  for (const auto& instance : cluster->instances) {
    VectorXf dist = instance - cluster->mean;
    for (int i = 0; i < instance.size(); ++i) {
      total_variance(i) += dist(i) * dist(i);
    }
  }
  total_variance = total_variance / cluster->instances.size();
  for (int i = 0; i < total_variance.size(); ++i) {
    if (total_variance[i] < gm_config_.min_variance()) {
      total_variance[i] = gm_config_.min_variance();
    }
  }
  cluster->variance = total_variance;
}

GaussianCluster GaussianMemory::ConvertToGaussianCluster(
    const InMemoryClusterData& cluster) {
  GaussianCluster gaussian_cluster;
  gaussian_cluster.mutable_mean()->mutable_value()->Reserve(
      cluster.mean.size());
  gaussian_cluster.mutable_variance()->mutable_value()->Reserve(
      cluster.variance.size());
  for (int i = 0; i < cluster.mean.size(); ++i) {
    gaussian_cluster.mutable_mean()->add_value(cluster.mean(i));
    gaussian_cluster.mutable_variance()->add_value(cluster.variance(i));
  }
  return gaussian_cluster;
}

GaussianMemoryCheckpointMetaData GaussianMemory::ConvertToCheckpointMetaData() {
  GaussianMemoryCheckpointMetaData result;
  absl::MutexLock l(&mu_);
  for (const auto& in_memory_cluster : cluster_list_) {
    auto* cluster_data = result.add_cluster_data();
    *cluster_data->mutable_gaussian_cluster() =
        ConvertToGaussianCluster(in_memory_cluster);
    for (const auto& instance : in_memory_cluster.instances) {
      auto* proto_instance = cluster_data->add_activation_instance();
      for (int i = 0; i < instance.size(); ++i) {
        proto_instance->add_value(instance[i]);
      }
    }
  }
  return result;
}

absl::Status GaussianMemory::ImportInternal(const std::string& saved_path) {
  GaussianMemoryCheckpointMetaData meta_data;
  RET_CHECK_OK(ReadTextProto(saved_path, &meta_data));
  RET_CHECK_TRUE(meta_data.cluster_data_size() <= gm_config_.max_num_clusters())
      << "Too many clusters.";
  absl::MutexLock l(&mu_);
  cluster_list_.clear();
  for (const auto& cluster_data : meta_data.cluster_data()) {
    InMemoryClusterData in_mem_data;
    in_mem_data.mean = std::move(
        ToInMemoryEmbeddingVector(cluster_data.gaussian_cluster().mean()).vec);

    in_mem_data.variance = std::move(
        ToInMemoryEmbeddingVector(cluster_data.gaussian_cluster().variance())
            .vec);
    for (int i = 0; i < in_mem_data.variance.size(); ++i) {
      RET_CHECK_TRUE(in_mem_data.variance(i) >= gm_config_.min_variance())
          << "Too small variance value.";
    }

    for (const auto& instance : cluster_data.activation_instance()) {
      in_mem_data.instances.push_back(
          std::move(ToInMemoryEmbeddingVector(instance).vec));
    }
    cluster_list_.push_back(std::move(in_mem_data));
  }
  cluster_counter_ = cluster_list_.size();
  return absl::OkStatus();
}

}  // namespace memory_store
}  // namespace carls
