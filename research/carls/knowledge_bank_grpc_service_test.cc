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
#include "research/carls/base/proto_helper.h"
#include "research/carls/testing/test_helper.h"

namespace carls {
namespace {

using ::grpc::ServerContext;

}  // namespace

class KnowledgeBankGrpcServiceImplTest : public ::testing::Test {
 protected:
  KnowledgeBankGrpcServiceImplTest()
      : de_config_(ParseTextProtoOrDie<DynamicEmbeddingConfig>(R"(
          embedding_dimension: 2
          knowledge_bank_config {
            initializer { zero_initializer {} }
            extension {
              [type.googleapis.com/carls.InProtoKnowledgeBankConfig] {}
            }
          }
        )")) {}

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
  EXPECT_OK(kbs_server_.StartSession(&context_, &request, &response));
  EXPECT_TRUE(!response.session_handle().empty());
  EXPECT_EQ(1, kbs_server_.KnowledgeBankSize());

  // Another valid config
  request.set_name("another embedding");
  EXPECT_OK(kbs_server_.StartSession(&context_, &request, &response));
  EXPECT_TRUE(!response.session_handle().empty());
  EXPECT_EQ(2, kbs_server_.KnowledgeBankSize());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, Lookup_EmptyInput) {
  // Starts a valid session.
  StartSessionRequest start_request;
  StartSessionResponse start_response;
  start_request.set_name("emb1");
  *start_request.mutable_config() = de_config_;
  ASSERT_TRUE(
      kbs_server_.StartSession(&context_, &start_request, &start_response)
          .ok());
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
  ASSERT_TRUE(
      kbs_server_.StartSession(&context_, &start_request, &start_response)
          .ok());
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Tests Lookup RPC.
  LookupRequest request;
  LookupResponse response;
  request.set_session_handle(session_handle);
  request.set_update(true);
  request.add_key("key1");
  EXPECT_OK(kbs_server_.Lookup(&context_, &request, &response));
  EXPECT_THAT(response, EqualsProto<LookupResponse>(R"(
                embedding_table {
                  key: "key1"
                  value { tag: "key1" value: 0 value: 0 weight: 1 }
                }
              )"));

  // Multiple keys.
  request.add_key("key2");
  ASSERT_TRUE(kbs_server_.Lookup(&context_, &request, &response).ok());
  LookupResponse expected_response = ParseTextProtoOrDie<LookupResponse>(R"(
    embedding_table {
      key: "key1"
      value { tag: "key1" value: 0 value: 0 weight: 2 }
    }
    embedding_table {
      key: "key2"
      value { tag: "key2" value: 0 value: 0 weight: 1 }
    }
  )");
  ASSERT_EQ(2, response.embedding_table().size());
  ASSERT_TRUE(response.embedding_table().contains("key1"));
  ASSERT_TRUE(response.embedding_table().contains("key2"));
  EXPECT_THAT(response.embedding_table().at("key1"),
              EqualsProto(expected_response.embedding_table().at("key1")));
  EXPECT_THAT(response.embedding_table().at("key2"),
              EqualsProto(expected_response.embedding_table().at("key2")));

  // No update, valid keys.
  request.set_update(false);
  ASSERT_TRUE(kbs_server_.Lookup(&context_, &request, &response).ok());
  expected_response = ParseTextProtoOrDie<LookupResponse>(R"(
    embedding_table {
      key: "key1"
      value { tag: "key1" value: 0 value: 0 weight: 2 }
    }
    embedding_table {
      key: "key2"
      value { tag: "key2" value: 0 value: 0 weight: 1 }
    }
  )");
  ASSERT_EQ(2, response.embedding_table().size());
  ASSERT_TRUE(response.embedding_table().contains("key1"));
  ASSERT_TRUE(response.embedding_table().contains("key2"));
  EXPECT_THAT(response.embedding_table().at("key1"),
              EqualsProto(expected_response.embedding_table().at("key1")));
  EXPECT_THAT(response.embedding_table().at("key2"),
              EqualsProto(expected_response.embedding_table().at("key2")));

  // No update, invalid keys.
  request.add_key("oov");
  ASSERT_TRUE(kbs_server_.Lookup(&context_, &request, &response).ok());
  // Only 2 results returned.
  EXPECT_EQ(2, response.embedding_table().size());
}

TEST_F(KnowledgeBankGrpcServiceImplTest, Lookup_ColdStart) {
  StartSessionRequest start_request;
  start_request.set_name("emb1");
  *start_request.mutable_config() = de_config_;

  // Cold start without calling StartSession first, still works.
  KnowledgeBankGrpcServiceImpl kbs_server_new;
  LookupRequest request =
      ParseTextProtoOrDie<LookupRequest>(R"(
        update: true key: "key1" key: "key2"
      )");
  request.set_session_handle(start_request.SerializeAsString());
  LookupResponse response;
  ASSERT_TRUE(kbs_server_new.Lookup(&context_, &request, &response).ok());
  LookupResponse expected_response = ParseTextProtoOrDie<LookupResponse>(R"(
    embedding_table {
      key: "key1"
      value { tag: "key1" value: 0 value: 0 weight: 1 }
    }
    embedding_table {
      key: "key2"
      value { tag: "key2" value: 0 value: 0 weight: 1 }
    }
  )");
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
  ASSERT_TRUE(
      kbs_server_.StartSession(&context_, &start_request, &start_response)
          .ok());
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
  ASSERT_TRUE(
      kbs_server_.StartSession(&context_, &start_request, &start_response)
          .ok());
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Update a single key.
  UpdateRequest update_request;
  UpdateResponse update_response;
  update_request.set_session_handle(session_handle);
  (*update_request.mutable_values())["key1"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"(
        value: 1 value: 2
      )");
  EXPECT_OK(kbs_server_.Update(&context_, &update_request, &update_response));

  // Check result.
  LookupRequest lookup_request;
  LookupResponse lookup_response;
  lookup_request.set_session_handle(session_handle);
  lookup_request.set_update(true);
  lookup_request.add_key("key1");
  EXPECT_OK(kbs_server_.Lookup(&context_, &lookup_request, &lookup_response));
  EXPECT_THAT(lookup_response, EqualsProto<LookupResponse>(R"(
                embedding_table {
                  key: "key1"
                  value { value: 1 value: 2 weight: 1 }
                }
              )"));

  // Update multiple keys.
  update_request.clear_values();
  (*update_request.mutable_values())["key2"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"(
        value: 3 value: 4
      )");
  EXPECT_OK(kbs_server_.Update(&context_, &update_request, &update_response));

  // Check results.
  lookup_request.add_key("key2");
  ASSERT_TRUE(
      kbs_server_.Lookup(&context_, &lookup_request, &lookup_response).ok());
  LookupResponse expected_response = ParseTextProtoOrDie<LookupResponse>(R"(
    embedding_table {
      key: "key1"
      value { value: 1 value: 2 weight: 2 }
    }
    embedding_table {
      key: "key2"
      value { value: 3 value: 4 weight: 1 }
    }
  )");
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
  ASSERT_TRUE(
      kbs_server_.StartSession(&context_, &start_request, &start_response)
          .ok());
  ASSERT_TRUE(!start_response.session_handle().empty());
  const auto& session_handle = start_response.session_handle();

  // Update the gradient of a non-existential key.
  UpdateRequest update_request;
  UpdateResponse update_response;
  update_request.set_session_handle(session_handle);
  (*update_request.mutable_gradients())["key1"] =
      ParseTextProtoOrDie<EmbeddingVectorProto>(R"(
        value: 0.1 value: 0.2
      )");
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
  EXPECT_OK(kbs_server_.Lookup(&context_, &lookup_request, &lookup_response));
  EXPECT_THAT(lookup_response, EqualsProto<LookupResponse>(R"(
                embedding_table {
                  key: "key1"
                  value { tag: "key1" value: 0 value: 0 weight: 1 }
                }
              )"));

  // Now update the gradients.
  status = kbs_server_.Update(&context_, &update_request, &update_response);
  ASSERT_TRUE(status.ok());

  // Check results.
  ASSERT_TRUE(
      kbs_server_.Lookup(&context_, &lookup_request, &lookup_response).ok());
  ASSERT_EQ(1, lookup_response.embedding_table_size());
  ASSERT_TRUE(lookup_response.embedding_table().contains("key1"));
  const auto embed = lookup_response.embedding_table().at("key1");
  EXPECT_EQ("key1", embed.tag());
  ASSERT_EQ(2, embed.value_size());
  EXPECT_FLOAT_EQ(-0.01, embed.value(0));
  EXPECT_FLOAT_EQ(-0.02, embed.value(1));
  EXPECT_FLOAT_EQ(2, embed.weight());
}

}  // namespace carls
