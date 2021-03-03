# CARLS: Cross-platform Asynchronous Representation Learning System

## Components Overview

*   TensorFlow API: Python functions that are implemented by C++ ops.
*   KnowledgeBank Service (KBS): a gRPC server that implements embedding
    lookup/update.
*   KnowledgeBank Manager: client side C++ hub that talks to KBS.
*   Storage System: underlying storage for Knowledge Bank, e.g.,
    InProtoKnowledgeBank for in-memory storage.

![](g3doc/images/knowledge_bank_server.png)
