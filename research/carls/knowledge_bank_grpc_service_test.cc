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

#include "research/carls/knowledge_bank_grpc_service.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "research/carls/base/file_helper.h"
#include "research/carls/base/proto_helper.h"
#include "research/carls/embedding.pb.h"  // proto to pb
#include "research/carls/knowledge_bank_service.pb.h"  // proto to pb
#include "research/carls/testing/test_helper.h"

namespace carls {
namespace {

using candidate_sampling::BruteForceTopkSamplerConfig;
using candidate_sampling::LogUniformSamplerConfig;
using candidate_sampling::SampledResult;
using ::grpc::ServerContext;

}  // namespace

class KnowledgeBankGrpcServiceImplTest : public ::testing::Test {
 protected:
  KnowledgeBankGrpcServiceImplTest()
      : de_config_(ParseTextProtoOrDie<DynamicEmbeddingConfig>(R"pb(
          embedding_dimension: 2
          knowledge_bank_config {
            initializer { zero_initializer {} }
            extension {
              [type.googleapis.com/carls.InProtoKnowledgeBankConfig] {}
            }
          }
        )pb")) {}

  // A dummy ServerContext for tests.
  ServerContext context_;

  // An instance of the service under test.
  KnowledgeBankGrpcServiceImpl kbs_server_;

  // A simple valid config.
  DynamicEmbeddingConfig de_config_;
};

TEST_F(KnowledgeBankGrpcServiceImplTest, StartSession_EmptyInput) {
  StartSessionRequest request;
  StartSessionResponse response;
  auto status = kbs_server_.StartSession(&context_, &request, &response);
  EXPECT_EQ("Name is empty.", status.error_message());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, StartSession_EmptyConfig) {
  StartSessionRequest request;
  StartSessionResponse response;
  request.set_name("an embedding");
  auto status = kbs_server_.StartSession(&context_, &request, &response);
  EXPECT_EQ("Creating KnowledgeBank failed.", status.error_message());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, StartSession_ValidConfigs) {
  StartSessionRequest request;
  StartSessionResponse response;
  // A valid config
  request.set_name("an embedding");
  *request.mutable_config() = de_config_;
  ASSERT_OK(kbs_server_.StartSession(&context_, &request, &response));
  EXPECT_TRUE(!response.session_handle().empty());
  EXPECT_EQ(1, kbs_server_.KnowledgeBankSize());

  // Another valid config
  request.set_name("another embedding");
  ASSERT_OK(kbs_server_.StartSession(&context_, &request, &response));
  EXPECT_TRUE(!response.session_handle().empty());
  EXPECT_EQ(2, kbs_server_.KnowledgeBankSize());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, Lookup_EmptyInput) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  *start_request.mutable_config() = de_config_;
  ASSERT_OK(
      kbs_server_.StartSession(&context_, &start_request, &start_response));
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Tests Lookup RPC.
  LookupRequest request;
  LookupResponse response;
  request.set_session_handle(session_handle);
  request.set_update(true);
  auto status = kbs_server_.Lookup(&context_, &request, &response);
  EXPECT_EQ("Empty input keys.", status.error_message());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, Lookup_ValidInput) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  *start_request.mutable_config() = de_config_;
  ASSERT_OK(
      kbs_server_.StartSession(&context_, &start_request, &start_response));
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Tests Lookup RPC.
  LookupRequest request;
  LookupResponse response;
  request.set_session_handle(session_handle);
  request.set_update(true);
  request.add_key("key1");
  ASSERT_OK(kbs_server_.Lookup(&context_, &request, &response));
  EXPECT_THAT(response, EqualsProto<LookupResponse>(R"pb(
                embedding_table {
                  key: "key1"
                  value { tag: "key1" value: 0 value: 0 weight: 1 }
                }
              )pb"));

  // Multiple keys.
  request.add_key("key2");
  ASSERT_OK(kbs_server_.Lookup(&context_, &request, &response));
  LookupResponse expected_response = ParseTextProtoOrDie<LookupResponse>(R"pb(
    embedding_table {
      key: "key1"
      value { tag: "key1" value: 0 value: 0 weight: 2 }
    }
    embedding_table {
      key: "key2"
      value { tag: "key2" value: 0 value: 0 weight: 1 }
    }
  )pb");
  ASSERT_EQ(2, response.embedding_table().size());
  ASSERT_TRUE(response.embedding_table().contains("key1"));
  ASSERT_TRUE(response.embedding_table().contains("key2"));
  EXPECT_THAT(response.embedding_table().at("key1"),
              EqualsProto(expected_response.embedding_table().at("key1")));
  EXPECT_THAT(response.embedding_table().at("key2"),
              EqualsProto(expected_response.embedding_table().at("key2")));

  // No update, valid keys.
  request.set_update(false);
  ASSERT_OK(kbs_server_.Lookup(&context_, &request, &response));
  expected_response = ParseTextProtoOrDie<LookupResponse>(R"pb(
    embedding_table {
      key: "key1"
      value { tag: "key1" value: 0 value: 0 weight: 2 }
    }
    embedding_table {
      key: "key2"
      value { tag: "key2" value: 0 value: 0 weight: 1 }
    }
  )pb");
  ASSERT_EQ(2, response.embedding_table().size());
  ASSERT_TRUE(response.embedding_table().contains("key1"));
  ASSERT_TRUE(response.embedding_table().contains("key2"));
  EXPECT_THAT(response.embedding_table().at("key1"),
              EqualsProto(expected_response.embedding_table().at("key1")));
  EXPECT_THAT(response.embedding_table().at("key2"),
              EqualsProto(expected_response.embedding_table().at("key2")));

  // No update, invalid keys.
  request.add_key("oov");
  ASSERT_OK(kbs_server_.Lookup(&context_, &request, &response));
  // Only 2 results returned.
  EXPECT_EQ(2, response.embedding_table().size());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, Lookup_ColdStart) {
  StartSessionRequest start_request;
  start_request.set_name("emb1");
  *start_request.mutable_config() = de_config_;

  // Cold start without calling StartSession first, still works.
  KnowledgeBankGrpcServiceImpl kbs_server_new;
  LookupRequest request = ParseTextProtoOrDie<LookupRequest>(R"pb(
    update: true
    key: "key1"
    key: "key2"
  )pb");
  request.set_session_handle(start_request.SerializeAsString());
  LookupResponse response;
  ASSERT_OK(kbs_server_new.Lookup(&context_, &request, &response));
  LookupResponse expected_response = ParseTextProtoOrDie<LookupResponse>(R"pb(
    embedding_table {
      key: "key1"
      value { tag: "key1" value: 0 value: 0 weight: 1 }
    }
    embedding_table {
      key: "key2"
      value { tag: "key2" value: 0 value: 0 weight: 1 }
    }
  )pb");
  ASSERT_EQ(2, response.embedding_table().size());
  ASSERT_TRUE(response.embedding_table().contains("key1"));
  ASSERT_TRUE(response.embedding_table().contains("key2"));
  EXPECT_THAT(response.embedding_table().at("key1"),
              EqualsProto(expected_response.embedding_table().at("key1")));
  EXPECT_THAT(response.embedding_table().at("key2"),
              EqualsProto(expected_response.embedding_table().at("key2")));
}

TEST_F(KnowledgeBankGrpcServiceImplTest, Update_EmptyInput) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  *start_request.mutable_config() = de_config_;
  ASSERT_OK(
      kbs_server_.StartSession(&context_, &start_request, &start_response));
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Missing session_handle.
  UpdateRequest request;
  UpdateResponse response;
  auto status = kbs_server_.Update(&context_, &request, &response);
  EXPECT_EQ("session_handle is empty.", status.error_message());

  // Empty input.
  request.set_session_handle(session_handle);
  status = kbs_server_.Update(&context_, &request, &response);
  EXPECT_EQ("input is empty.", status.error_message());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, UpdateEmbedding) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  *start_request.mutable_config() = de_config_;
  ASSERT_OK(
      kbs_server_.StartSession(&context_, &start_request, &start_response));
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Update a single key.
  UpdateRequest update_request;
  UpdateResponse update_response;
  update_request.set_session_handle(session_handle);
  (*update_request.mutable_values())["key1"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 1 value: 2
      )pb");
  ASSERT_OK(kbs_server_.Update(&context_, &update_request, &update_response));

  // Check result.
  LookupRequest lookup_request;
  LookupResponse lookup_response;
  lookup_request.set_session_handle(session_handle);
  lookup_request.set_update(true);
  lookup_request.add_key("key1");
  ASSERT_OK(kbs_server_.Lookup(&context_, &lookup_request, &lookup_response));
  EXPECT_THAT(lookup_response, EqualsProto<LookupResponse>(R"pb(
                embedding_table {
                  key: "key1"
                  value { value: 1 value: 2 weight: 1 }
                }
              )pb"));

  // Update multiple keys.
  update_request.clear_values();
  (*update_request.mutable_values())["key2"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 3 value: 4
      )pb");
  ASSERT_OK(kbs_server_.Update(&context_, &update_request, &update_response));

  // Check results.
  lookup_request.add_key("key2");
  ASSERT_OK(kbs_server_.Lookup(&context_, &lookup_request, &lookup_response));
  LookupResponse expected_response = ParseTextProtoOrDie<LookupResponse>(R"pb(
    embedding_table {
      key: "key1"
      value { value: 1 value: 2 weight: 2 }
    }
    embedding_table {
      key: "key2"
      value { value: 3 value: 4 weight: 1 }
    }
  )pb");
  ASSERT_EQ(2, lookup_response.embedding_table().size());
  ASSERT_TRUE(lookup_response.embedding_table().contains("key1"));
  ASSERT_TRUE(lookup_response.embedding_table().contains("key2"));
  EXPECT_THAT(lookup_response.embedding_table().at("key1"),
              EqualsProto(expected_response.embedding_table().at("key1")));
  EXPECT_THAT(lookup_response.embedding_table().at("key2"),
              EqualsProto(expected_response.embedding_table().at("key2")));
}

TEST_F(KnowledgeBankGrpcServiceImplTest, UpdateGradient) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  de_config_.mutable_gradient_descent_config()->set_learning_rate(0.1);
  de_config_.mutable_gradient_descent_config()->mutable_sgd();
  *start_request.mutable_config() = de_config_;
  ASSERT_OK(
      kbs_server_.StartSession(&context_, &start_request, &start_response));
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Update the gradient of a non-existential key.
  UpdateRequest update_request;
  UpdateResponse update_response;
  update_request.set_session_handle(session_handle);
  (*update_request.mutable_gradients())["key1"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 0.1 value: 0.2
      )pb");
  auto status =
      kbs_server_.Update(&context_, &update_request, &update_response);
  ASSERT_FALSE(status.ok());
  EXPECT_EQ("No valid keys for gradient update.", status.error_message());

  // Add embedding into the store.
  LookupRequest lookup_request;
  LookupResponse lookup_response;
  lookup_request.set_session_handle(session_handle);
  lookup_request.set_update(true);
  lookup_request.add_key("key1");
  ASSERT_OK(kbs_server_.Lookup(&context_, &lookup_request, &lookup_response));
  EXPECT_THAT(lookup_response, EqualsProto<LookupResponse>(R"pb(
                embedding_table {
                  key: "key1"
                  value { tag: "key1" value: 0 value: 0 weight: 1 }
                }
              )pb"));

  // Now update the gradients.
  ASSERT_OK(kbs_server_.Update(&context_, &update_request, &update_response));

  // Check results.
  ASSERT_OK(kbs_server_.Lookup(&context_, &lookup_request, &lookup_response));
  ASSERT_EQ(1, lookup_response.embedding_table_size());
  ASSERT_TRUE(lookup_response.embedding_table().contains("key1"));
  const auto embed = lookup_response.embedding_table().at("key1");
  EXPECT_EQ("key1", embed.tag());
  ASSERT_EQ(2, embed.value_size());
  EXPECT_FLOAT_EQ(-0.01, embed.value(0));
  EXPECT_FLOAT_EQ(-0.02, embed.value(1));
  EXPECT_FLOAT_EQ(2, embed.weight());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, Sample_BruteForceTopK) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  BruteForceTopkSamplerConfig topk_config;
  topk_config.set_similarity_type(candidate_sampling::DOT_PRODUCT);
  de_config_.mutable_candidate_sampler_config()->mutable_extension()->PackFrom(
      topk_config);
  *start_request.mutable_config() = de_config_;
  ASSERT_OK(
      kbs_server_.StartSession(&context_, &start_request, &start_response));
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Update a few keys.
  UpdateRequest update_request;
  UpdateResponse update_response;
  update_request.set_session_handle(session_handle);
  (*update_request.mutable_values())["key1"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 1 value: 2
      )pb");
  (*update_request.mutable_values())["key2"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 3 value: 4
      )pb");
  (*update_request.mutable_values())["key3"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 5 value: 6
      )pb");
  ASSERT_OK(kbs_server_.Update(&context_, &update_request, &update_response));

  // TopK sampling.
  SampleRequest sample_request;
  SampleResponse sample_response;
  sample_request.set_session_handle(session_handle);
  sample_request.set_num_samples(2);
  (*sample_request.add_sample_context()->mutable_activation()) =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 1 value: 2
      )pb");
  ASSERT_OK(kbs_server_.Sample(&context_, &sample_request, &sample_response));
  EXPECT_THAT(sample_response, EqualsProto<SampleResponse>(R"pb(
                samples {
                  sampled_result {
                    topk_sampling_result {
                      key: "key3"
                      embedding { value: 5 value: 6 }
                      similarity: 17
                    }
                  }
                  sampled_result {
                    topk_sampling_result {
                      key: "key2"
                      embedding { value: 3 value: 4 }
                      similarity: 11
                    }
                  }
                }
              )pb"));
}

TEST_F(KnowledgeBankGrpcServiceImplTest, Sample_LogUniformSample) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  LogUniformSamplerConfig neg_sample_config;
  neg_sample_config.set_unique(true);
  de_config_.mutable_candidate_sampler_config()->mutable_extension()->PackFrom(
      neg_sample_config);
  *start_request.mutable_config() = de_config_;
  ASSERT_OK(
      kbs_server_.StartSession(&context_, &start_request, &start_response));
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Update a few keys.
  UpdateRequest update_request;
  UpdateResponse update_response;
  update_request.set_session_handle(session_handle);
  (*update_request.mutable_values())["key1"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 1 value: 2
      )pb");
  (*update_request.mutable_values())["key2"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 3 value: 4
      )pb");
  (*update_request.mutable_values())["key3"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 5 value: 6
      )pb");
  ASSERT_OK(kbs_server_.Update(&context_, &update_request, &update_response));

  // Sample without update, all available keys {"key1", "key2", "key3"}
  SampleRequest sample_request;
  SampleResponse sample_response;
  sample_request.set_session_handle(session_handle);
  sample_request.set_num_samples(3);
  sample_request.add_sample_context()->add_positive_key("key1");
  ASSERT_OK(kbs_server_.Sample(&context_, &sample_request, &sample_response));
  ASSERT_EQ(1, sample_response.samples_size());
  auto comparator = [](const SampledResult& lhs,
                       const SampledResult& rhs) -> bool {
    return lhs.negative_sampling_result().key() <
           rhs.negative_sampling_result().key();
  };
  std::sort(
      sample_response.mutable_samples(0)->mutable_sampled_result()->begin(),
      sample_response.mutable_samples(0)->mutable_sampled_result()->end(),
      comparator);
  EXPECT_THAT(sample_response, EqualsProto<SampleResponse>(R"pb(
                samples {
                  sampled_result {
                    negative_sampling_result {
                      key: "key1"
                      embedding { value: 1 value: 2 }
                      is_positive: true
                      expected_count: 1
                    }
                  }
                  sampled_result {
                    negative_sampling_result {
                      key: "key2"
                      embedding { value: 3 value: 4 }
                      expected_count: 1
                    }
                  }
                  sampled_result {
                    negative_sampling_result {
                      key: "key3"
                      embedding { value: 5 value: 6 }
                      expected_count: 1
                    }
                  }
                }
              )pb"));

  // Sample with update, "key4" is added to the knowledge bank.
  sample_request.mutable_sample_context(0)->add_positive_key("key4");
  sample_request.set_num_samples(4);
  sample_request.set_update(true);
  sample_response.Clear();
  ASSERT_OK(kbs_server_.Sample(&context_, &sample_request, &sample_response));
  ASSERT_EQ(1, sample_response.samples_size());
  std::sort(
      sample_response.mutable_samples(0)->mutable_sampled_result()->begin(),
      sample_response.mutable_samples(0)->mutable_sampled_result()->end(),
      comparator);
  // "key1", "key2" and "key3" are updated through Update() so their weights are
  // zero. "key4" is updated through LookupWithUpdate() so its weight/frequency
  // is 1.
  EXPECT_THAT(sample_response, EqualsProto<SampleResponse>(R"pb(
                samples {
                  sampled_result {
                    negative_sampling_result {
                      key: "key1"
                      embedding { value: 1 value: 2 }
                      is_positive: true
                      expected_count: 1
                    }
                  }
                  sampled_result {
                    negative_sampling_result {
                      key: "key2"
                      embedding { value: 3 value: 4}
                      expected_count: 1
                    }
                  }
                  sampled_result {
                    negative_sampling_result {
                      key: "key3"
                      embedding { value: 5 value: 6 }
                      expected_count: 1
                    }
                  }
                  sampled_result {
                    negative_sampling_result {
                      key: "key4"
                      embedding { tag: "key4" value: 0 value: 0 weight: 1 }
                      is_positive: true
                      expected_count: 1
                    }
                  }
                }
              )pb"));
}

TEST_F(KnowledgeBankGrpcServiceImplTest, ExportAndImport) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  *start_request.mutable_config() = de_config_;
  ASSERT_OK(
      kbs_server_.StartSession(&context_, &start_request, &start_response));
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Add two embeddings to knowledge bank.
  LookupRequest lookup_request;
  LookupResponse lookup_response;
  lookup_request.set_session_handle(session_handle);
  lookup_request.set_update(true);
  lookup_request.add_key("key1");
  lookup_request.add_key("key2");
  ASSERT_OK(kbs_server_.Lookup(&context_, &lookup_request, &lookup_response));
  ASSERT_EQ(2, lookup_response.embedding_table().size());

  // Export current knowledge bank.
  ExportRequest export_request;
  ExportResponse export_response;
  export_request.set_session_handle(session_handle);
  export_request.set_export_directory(testing::TempDir());
  ASSERT_OK(kbs_server_.Export(&context_, &export_request, &export_response));
  const std::string expected_path =
      JoinPath(testing::TempDir(), "emb1/embedding_store_meta_data.pbtxt");
  EXPECT_THAT(export_response,
              EqualsProto<ExportResponse>(absl::StrFormat(
                  "knowledge_bank_saved_path: '%s'", expected_path)));

  // Now updates existing and additional keys
  UpdateRequest update_request;
  UpdateResponse update_response;
  update_request.set_session_handle(session_handle);
  (*update_request.mutable_values())["key1"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 1 value: 2
      )pb");
  (*update_request.mutable_values())["key2"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 3 value: 4
      )pb");
  (*update_request.mutable_values())["key3"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"pb(
        value: 5 value: 6
      )pb");
  ASSERT_OK(kbs_server_.Update(&context_, &update_request, &update_response));

  // Restores previous knowledge bank.
  ImportRequest import_request;
  ImportResponse import_response;
  import_request.set_session_handle(session_handle);
  import_request.set_knowledge_bank_saved_path(
      export_response.knowledge_bank_saved_path());
  ASSERT_OK(kbs_server_.Import(&context_, &import_request, &import_response));

  // Now checks the results is restored.
  ASSERT_OK(kbs_server_.Lookup(&context_, &lookup_request, &lookup_response));
  ASSERT_EQ(2, lookup_response.embedding_table().size());
  ASSERT_TRUE(lookup_response.embedding_table().contains("key1"));
  ASSERT_TRUE(lookup_response.embedding_table().contains("key2"));
  EXPECT_THAT(lookup_response.embedding_table().at("key1"),
              EqualsProto<EmbeddingVectorProto>(
                  "tag: 'key1' value: 0 value: 0 weight: 2"));
  EXPECT_THAT(lookup_response.embedding_table().at("key2"),
              EqualsProto<EmbeddingVectorProto>(
                  "tag: 'key2' value: 0 value: 0 weight: 2"));
}

}  // namespace carls
