/*Copyright 2020 Google LLC

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

#include "research/carls/dynamic_embedding_manager.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/candidate_sampling/candidate_sampler_config.pb.h"  // proto to pb
#include "research/carls/kbs_server_helper.h"
#include "research/carls/testing/test_helper.h"
#include "tensorflow/core/framework/types.pb.h"  // proto to pb

namespace carls {

using ::tensorflow::Tensor;
using ::tensorflow::TensorShape;
using ::tensorflow::tstring;

class DynamicEmbeddingManagerTest : public ::testing::Test {
 protected:
  DynamicEmbeddingManagerTest() {}

  DynamicEmbeddingConfig BuildConfig(const int dimension,
                                     const float learning_rate = 0.1f) {
    return ParseTextProtoOrDie<DynamicEmbeddingConfig>(
        absl::StrFormat(R"(
        embedding_dimension: %d
        knowledge_bank_config {
          initializer { zero_initializer {} }
          extension {
            [type.googleapis.com/carls.InProtoKnowledgeBankConfig] {}
          }
        }
        gradient_descent_config {
          learning_rate: %f
          sgd {}
        }
      )",
                        dimension, learning_rate));
  }
};

TEST_F(DynamicEmbeddingManagerTest, Create) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  const std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/10);
  // Empty address.
  EXPECT_TRUE(DynamicEmbeddingManager::Create(config, "emb",
                                              /*kbs_address=*/"") == nullptr);
  // Invalid config.
  DynamicEmbeddingConfig empty_config;
  EXPECT_TRUE(DynamicEmbeddingManager::Create(empty_config, "emb",
                                              /*kbs_address=*/"") == nullptr);
  // A valid case.
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  EXPECT_TRUE(de_manager != nullptr);
}

TEST_F(DynamicEmbeddingManagerTest, Lookup_EmptyInput) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  const std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/10);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  Tensor empty_keys;
  Tensor output;
  auto status = de_manager->Lookup(empty_keys, /*update=*/true, &output);
  EXPECT_EQ("No input.", status.message());
}

TEST_F(DynamicEmbeddingManagerTest, Lookup_1DInput) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/2);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  Tensor keys(tensorflow::DT_STRING, TensorShape({2}));
  auto keys_value = keys.vec<tstring>();
  keys_value(0) = "first";
  keys_value(1) = "";
  Tensor output(tensorflow::DT_FLOAT, TensorShape({2, 2}));
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/true, &output).ok());
  auto output_values = output.matrix<float>();
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_FLOAT_EQ(0, output_values(i, j));
    }
  }
}

TEST_F(DynamicEmbeddingManagerTest, Lookup_2DInput) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/2);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  Tensor keys(tensorflow::DT_STRING, TensorShape({2, 2}));
  auto keys_value = keys.matrix<tstring>();
  keys_value(0, 0) = "first";
  keys_value(0, 1) = "second";
  keys_value(1, 0) = "third";
  keys_value(1, 1) = "";
  Tensor output = Tensor(tensorflow::DT_FLOAT, TensorShape({2, 2, 2}));
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/true, &output).ok());
  auto output_values = output.tensor<float, 3>();
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        EXPECT_FLOAT_EQ(0, output_values(i, j, k));
      }
    }
  }
}

TEST_F(DynamicEmbeddingManagerTest, UpdateValues_InvalidInputs) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/2);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  // Empty input.
  Tensor keys;
  Tensor values;
  EXPECT_ERROR_CONTAIN(de_manager->UpdateValues(keys, values),
                       "CheckInputForUpdate(keys, values)");

  // Inconsistent key size and value size.
  keys = Tensor(tensorflow::DT_STRING, TensorShape({3}));
  values = Tensor(tensorflow::DT_STRING, TensorShape({2, 2}));
  EXPECT_ERROR_CONTAIN(de_manager->UpdateValues(keys, values),
                       "CheckInputForUpdate(keys, values)");

  // Inconsistent embedding dimension.
  values = Tensor(tensorflow::DT_STRING, TensorShape({3, 4}));
  EXPECT_ERROR_CONTAIN(de_manager->UpdateValues(keys, values),
                       "CheckInputForUpdate(keys, values)");
}

TEST_F(DynamicEmbeddingManagerTest, UpdateValues_1DInput) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/2);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  Tensor keys(tensorflow::DT_STRING, TensorShape({3}));
  Tensor embed(tensorflow::DT_FLOAT, TensorShape({3, 2}));
  auto keys_value = keys.vec<tstring>();
  keys_value(0) = "first";
  keys_value(1) = "second";
  keys_value(2) = "third";
  auto embed_value = embed.matrix<float>();
  embed_value(0, 0) = -1;
  embed_value(0, 1) = 3;
  embed_value(1, 0) = 2;
  embed_value(1, 1) = -10;
  embed_value(2, 0) = -5;
  embed_value(2, 1) = 1;
  ASSERT_TRUE(de_manager->UpdateValues(keys, embed).ok());

  // Check results.
  Tensor output = Tensor(tensorflow::DT_FLOAT, TensorShape({3, 2}));
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/false, &output).ok());
  auto output_values = output.matrix<float>();
  EXPECT_FLOAT_EQ(-1, output_values(0, 0));
  EXPECT_FLOAT_EQ(3, output_values(0, 1));
  EXPECT_FLOAT_EQ(2, output_values(1, 0));
  EXPECT_FLOAT_EQ(-10, output_values(1, 1));
  EXPECT_FLOAT_EQ(-5, output_values(2, 0));
  EXPECT_FLOAT_EQ(1, output_values(2, 1));
}

TEST_F(DynamicEmbeddingManagerTest, UpdateValues_2DInput) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/2);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  Tensor keys(tensorflow::DT_STRING, TensorShape({2, 2}));
  Tensor embed(tensorflow::DT_FLOAT, TensorShape({2, 2, 2}));
  auto keys_value = keys.matrix<tstring>();
  keys_value(0, 0) = "first";
  keys_value(0, 1) = "second";
  keys_value(1, 0) = "third";
  keys_value(1, 1) = "";
  auto embed_values = embed.tensor<float, 3>();
  int val = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        embed_values(i, j, k) = val++;
      }
    }
  }
  ASSERT_TRUE(de_manager->UpdateValues(keys, embed).ok());

  // Check results.
  Tensor output = Tensor(tensorflow::DT_FLOAT, TensorShape({2, 2, 2}));
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/false, &output).ok());
  auto output_values = output.tensor<float, 3>();
  EXPECT_FLOAT_EQ(0, output_values(0, 0, 0));
  EXPECT_FLOAT_EQ(1, output_values(0, 0, 1));
  EXPECT_FLOAT_EQ(2, output_values(0, 1, 0));
  EXPECT_FLOAT_EQ(3, output_values(0, 1, 1));
  EXPECT_FLOAT_EQ(4, output_values(1, 0, 0));
  EXPECT_FLOAT_EQ(5, output_values(1, 0, 1));
  // For empty input, it returns all zeros.
  EXPECT_FLOAT_EQ(0, output_values(1, 1, 0));
  EXPECT_FLOAT_EQ(0, output_values(1, 1, 1));
}

TEST_F(DynamicEmbeddingManagerTest, UpdateGradients_1DInput) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  const std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/2);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  Tensor keys(tensorflow::DT_STRING, TensorShape({3}));
  auto keys_value = keys.vec<tstring>();
  keys_value(0) = "first";
  keys_value(1) = "second";
  keys_value(2) = "third";
  // Initial update returns all zeros.
  Tensor embed = Tensor(tensorflow::DT_FLOAT, TensorShape({3, 2}));
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/true, &embed).ok());

  // Updates the gradients using SGD.
  Tensor grads(tensorflow::DT_FLOAT, TensorShape({3, 2}));
  auto grads_values = grads.matrix<float>();
  grads_values(0, 0) = 1;
  grads_values(0, 1) = 2;
  grads_values(1, 0) = 3;
  grads_values(1, 1) = 4;
  grads_values(2, 0) = 5;
  grads_values(2, 1) = 6;
  ASSERT_TRUE(de_manager->UpdateGradients(keys, grads).ok());

  // Check results with learning rate set to 0.1.
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/false, &embed).ok());
  auto embed_values = embed.matrix<float>();
  EXPECT_FLOAT_EQ(-0.1, embed_values(0, 0));
  EXPECT_FLOAT_EQ(-0.2, embed_values(0, 1));
  EXPECT_FLOAT_EQ(-0.3, embed_values(1, 0));
  EXPECT_FLOAT_EQ(-0.4, embed_values(1, 1));
  EXPECT_FLOAT_EQ(-0.5, embed_values(2, 0));
  EXPECT_FLOAT_EQ(-0.6, embed_values(2, 1));
}

TEST_F(DynamicEmbeddingManagerTest, UpdateGradients_2DInput) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/2);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  Tensor keys(tensorflow::DT_STRING, TensorShape({2, 2}));
  auto keys_value = keys.matrix<tstring>();
  keys_value(0, 0) = "first";
  keys_value(0, 1) = "second";
  keys_value(1, 0) = "third";
  keys_value(1, 1) = "";
  // Initial update returns all zeros.
  Tensor embed(tensorflow::DT_FLOAT, TensorShape({2, 2, 2}));
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/true, &embed).ok());

  Tensor grads(tensorflow::DT_FLOAT, TensorShape({2, 2, 2}));
  auto grads_values = grads.tensor<float, 3>();
  int val = 0;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        grads_values(i, j, k) = val++;
      }
    }
  }
  ASSERT_TRUE(de_manager->UpdateGradients(keys, grads).ok());

  // Check results with learning rate set to 0.1.
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/false, &embed).ok());
  auto embed_values = embed.tensor<float, 3>();
  EXPECT_FLOAT_EQ(0, embed_values(0, 0, 0));
  EXPECT_FLOAT_EQ(-0.1, embed_values(0, 0, 1));
  EXPECT_FLOAT_EQ(-0.2, embed_values(0, 1, 0));
  EXPECT_FLOAT_EQ(-0.3, embed_values(0, 1, 1));
  EXPECT_FLOAT_EQ(-0.4, embed_values(1, 0, 0));
  EXPECT_FLOAT_EQ(-0.5, embed_values(1, 0, 1));
  // For empty input, it returns all zeros.
  EXPECT_FLOAT_EQ(0, embed_values(1, 1, 0));
  EXPECT_FLOAT_EQ(0, embed_values(1, 1, 1));
}

TEST_F(DynamicEmbeddingManagerTest, NegativeSampling) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/3);
  candidate_sampling::NegativeSamplerConfig negative_sampler;
  negative_sampler.set_unique(true);
  negative_sampler.set_sampler(
      candidate_sampling::NegativeSamplerConfig::LOG_UNIFORM);
  auto* sampler_config = config.mutable_candidate_sampler_config();
  sampler_config->mutable_extension()->PackFrom(negative_sampler);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  // A batch of size two, one with positive keys {"key1", "key2"}, another with
  // {"key3"}.
  Tensor positive_keys(tensorflow::DT_STRING, TensorShape({2, 2}));
  auto pos_keys_value = positive_keys.matrix<tstring>();
  pos_keys_value(0, 0) = "key1";
  pos_keys_value(0, 1) = "key2";
  pos_keys_value(1, 0) = "key3";
  pos_keys_value(1, 1) = "";
  // Initial update returns all zeros.
  Tensor embed(tensorflow::DT_FLOAT, TensorShape({2, 2, 3}));
  auto embed_value = embed.tensor<float, 3>();
  // Embedding for "key1"
  embed_value(0, 0, 0) = 1;
  embed_value(0, 0, 1) = 1;
  embed_value(0, 0, 2) = 1;
  // Embedding for "key2"
  embed_value(0, 1, 0) = 2;
  embed_value(0, 1, 1) = 2;
  embed_value(0, 1, 2) = 2;
  // Embedding for "key3"
  embed_value(1, 0, 0) = 3;
  embed_value(1, 0, 1) = 3;
  embed_value(1, 0, 2) = 3;
  ASSERT_OK(de_manager->UpdateValues(positive_keys, embed));

  // Sets input activation.
  Tensor input(tensorflow::DT_FLOAT, TensorShape({2, 3}));
  auto input_value = input.matrix<float>();
  input_value(0, 0) = 1;
  input_value(0, 1) = 2;
  input_value(0, 2) = 1;
  input_value(1, 0) = 3;
  input_value(1, 1) = 4;
  input_value(1, 2) = 1;

  Tensor output_keys(tensorflow::DT_STRING, TensorShape({2, 3}));
  Tensor output_label(tensorflow::DT_FLOAT, TensorShape({2, 3}));
  Tensor output_expected_count(tensorflow::DT_FLOAT, TensorShape({2, 3}));
  Tensor output_mask(tensorflow::DT_FLOAT, TensorShape({2}));
  Tensor output_embed(tensorflow::DT_FLOAT, TensorShape({2, 3, 3}));
  ASSERT_OK(de_manager->NegativeSampling(
      positive_keys, input, /*num_samples=*/3, /*update=*/true, &output_keys,
      &output_label, &output_expected_count, &output_mask, &output_embed));

  // Now checks the results. For both entries, they should return
  // {"key1", "key2", "key3"} since they are the only available keys.
  auto keys_value = output_keys.matrix<tstring>();
  std::set<std::string> keys1;
  std::set<std::string> keys2;
  for (int i = 0; i < 3; ++i) {
    keys1.insert(keys_value(0, i));
    keys2.insert(keys_value(1, i));
  }
  for (const auto& key : std::vector<std::string>{"key1", "key2", "key3"}) {
    EXPECT_TRUE(keys1.find(key) != keys1.end());
    EXPECT_TRUE(keys2.find(key) != keys2.end());
  }

  // Checks label output. Entry one should return 1 negative keys and entry two
  // should return 2 negative keys.
  auto labels_value = output_label.matrix<float>();
  for (int i = 0; i < 3; ++i) {
    // Entry one.
    if (keys_value(0, i) == "key1" || keys_value(0, i) == "key2") {
      EXPECT_FLOAT_EQ(1, labels_value(0, i));
    } else {
      EXPECT_FLOAT_EQ(0, labels_value(0, i));
    }
    // Entry two.
    if (keys_value(1, i) == "key3") {
      EXPECT_FLOAT_EQ(1, labels_value(1, i));
    } else {
      EXPECT_FLOAT_EQ(0, labels_value(1, i));
    }
  }

  // Checks output_expected_count. Should be all one's in this case.
  auto prob_value = output_expected_count.matrix<float>();
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 3; ++i) {
      EXPECT_FLOAT_EQ(1, prob_value(b, i));
    }
  }

  // Checks output_mask. Should also be all one's.
  auto mask_value = output_mask.vec<float>();
  EXPECT_FLOAT_EQ(1, mask_value(0));
  EXPECT_FLOAT_EQ(1, mask_value(1));

  // Checks output_embed.
  auto output_embed_value = output_embed.tensor<float, 3>();
  for (int b = 0; b < 2; ++b) {
    for (int i = 0; i < 3; ++i) {
      float expected_value = 0;
      if (keys_value(b, i) == "key1") {
        expected_value = 1;
      } else if (keys_value(b, i) == "key2") {
        expected_value = 2;
      } else {  // "key3"
        expected_value = 3;
      }
      for (int d = 0; d < 3; ++d) {
        EXPECT_FLOAT_EQ(expected_value, output_embed_value(b, i, d));
      }
    }
  }
}

TEST_F(DynamicEmbeddingManagerTest, Topk) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/3);
  candidate_sampling::BruteForceTopkSamplerConfig topk_sampler;
  topk_sampler.set_similarity_type(candidate_sampling::DOT_PRODUCT);
  auto* sampler_config = config.mutable_candidate_sampler_config();
  sampler_config->mutable_extension()->PackFrom(topk_sampler);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  // Add a few keys into the server.
  Tensor keys(tensorflow::DT_STRING, TensorShape({3}));
  auto keys_value = keys.vec<tstring>();
  keys_value(0) = "key1";
  keys_value(1) = "key2";
  keys_value(2) = "key3";
  // Initial update returns all zeros.
  Tensor embed(tensorflow::DT_FLOAT, TensorShape({3, 3}));
  auto embed_value = embed.matrix<float>();
  // Embedding for "key1"
  embed_value(0, 0) = 1;
  embed_value(0, 1) = 1;
  embed_value(0, 2) = 1;
  // Embedding for "key2"
  embed_value(1, 0) = 2;
  embed_value(1, 1) = 2;
  embed_value(1, 2) = 2;
  // Embedding for "key3"
  embed_value(2, 0) = 3;
  embed_value(2, 1) = 3;
  embed_value(2, 2) = 3;
  ASSERT_OK(de_manager->UpdateValues(keys, embed));

  // Sets input activations: [[1, 2, 1], [-1, -2, 1]].
  Tensor input(tensorflow::DT_FLOAT, TensorShape({2, 3}));
  auto input_value = input.matrix<float>();
  input_value(0, 0) = 1;
  input_value(0, 1) = 2;
  input_value(0, 2) = 1;
  input_value(1, 0) = -1;
  input_value(1, 1) = -2;
  input_value(1, 2) = 1;

  Tensor output_keys(tensorflow::DT_STRING, TensorShape({2, 3}));
  Tensor output_logits(tensorflow::DT_FLOAT, TensorShape({2, 3}));
  ASSERT_OK(de_manager->TopK(input, /*k=*/3, &output_keys, &output_logits));

  // Checks topk output ([w, b] * [x, 1]).
  // Entry one: "key3" > "key2" > "key1"
  // - "key1": [1, 1, 1] * [1, 2, 1] = 4
  // - "key2": [2, 2, 2] * [1, 2, 1] = 8
  // - "key3": [3, 3, 3] * [1, 2, 1] = 12
  // Entry two: "key1" > "key2" > "key3"
  // - "key1": [1, 1, 1] * [-1, -2, 1] = -2
  // - "key2": [2, 2, 2] * [-1, -2, 1] = -4
  // - "key3": [3, 3, 3] * [-1, -2, 1] = -6
  auto topk_keys_value = output_keys.matrix<tstring>();
  EXPECT_EQ("key3", topk_keys_value(0, 0));
  EXPECT_EQ("key2", topk_keys_value(0, 1));
  EXPECT_EQ("key1", topk_keys_value(0, 2));
  EXPECT_EQ("key1", topk_keys_value(1, 0));
  EXPECT_EQ("key2", topk_keys_value(1, 1));
  EXPECT_EQ("key3", topk_keys_value(1, 2));

  auto logits_value = output_logits.matrix<float>();
  EXPECT_FLOAT_EQ(12, logits_value(0, 0));
  EXPECT_FLOAT_EQ(8, logits_value(0, 1));
  EXPECT_FLOAT_EQ(4, logits_value(0, 2));
  EXPECT_FLOAT_EQ(-2, logits_value(1, 0));
  EXPECT_FLOAT_EQ(-4, logits_value(1, 1));
  EXPECT_FLOAT_EQ(-6, logits_value(1, 2));
}

TEST_F(DynamicEmbeddingManagerTest, ImportAndExport) {
  KnowledgeBankServiceOptions options;
  KbsServerHelper helper(options);
  std::string address = absl::StrCat("localhost:", helper.port());
  DynamicEmbeddingConfig config = BuildConfig(/*dimension=*/2);
  auto de_manager = DynamicEmbeddingManager::Create(config, "emb", address);
  ASSERT_TRUE(de_manager != nullptr);

  // Add a few keys.
  Tensor keys(tensorflow::DT_STRING, TensorShape({3}));
  auto keys_value = keys.vec<tstring>();
  keys_value(0) = "first";
  keys_value(1) = "second";
  keys_value(2) = "third";
  // Initial update returns all zeros.
  Tensor embed = Tensor(tensorflow::DT_FLOAT, TensorShape({3, 2}));
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/true, &embed).ok());

  // Export.
  std::string exported_path;
  ASSERT_TRUE(de_manager->Export(testing::TempDir(), &exported_path).ok());
  EXPECT_EQ(JoinPath(testing::TempDir(), "emb/embedding_store_meta_data.pbtxt"),
            exported_path);

  // Update the embeddings of a few keys.
  Tensor new_embed(tensorflow::DT_FLOAT, TensorShape({3, 2}));
  auto new_embed_value = new_embed.matrix<float>();
  new_embed_value(0, 0) = 1;
  new_embed_value(0, 1) = 2;
  new_embed_value(1, 0) = 3;
  new_embed_value(1, 1) = 4;
  new_embed_value(2, 0) = 5;
  new_embed_value(2, 1) = 6;
  ASSERT_TRUE(de_manager->UpdateValues(keys, new_embed).ok());

  // Now restore to previous state.
  ASSERT_TRUE(de_manager->Import(exported_path).ok());

  // Checks the results.
  ASSERT_TRUE(de_manager->Lookup(keys, /*update=*/false, &new_embed).ok());
  auto embed_value = embed.matrix<float>();
  new_embed_value = new_embed.matrix<float>();
  EXPECT_EQ(embed_value(0, 0), new_embed_value(0, 0));
  EXPECT_EQ(embed_value(0, 1), new_embed_value(0, 1));
  EXPECT_EQ(embed_value(1, 0), new_embed_value(1, 0));
  EXPECT_EQ(embed_value(1, 1), new_embed_value(1, 1));
  EXPECT_EQ(embed_value(2, 0), new_embed_value(2, 0));
  EXPECT_EQ(embed_value(2, 1), new_embed_value(2, 1));
}

}  // namespace carls
