# go-light-rag

[![Go Reference](https://pkg.go.dev/badge/github.com/MegaGrindStone/go-light-rag.svg)](https://pkg.go.dev/github.com/MegaGrindStone/go-light-rag)
![CI](https://github.com/MegaGrindStone/go-light-rag/actions/workflows/ci.yaml/badge.svg)
[![Go Report Card](https://goreportcard.com/badge/github.com/MegaGrindStone/go-light-rag)](https://goreportcard.com/report/github.com/MegaGrindStone/go-light-rag)
[![codecov](https://codecov.io/gh/MegaGrindStone/go-light-rag/branch/main/graph/badge.svg)](https://codecov.io/gh/MegaGrindStone/go-light-rag)


A Go library implementation of [LightRAG](https://github.com/HKUDS/LightRAG) - an advanced Retrieval-Augmented Generation (RAG) system that uniquely combines vector databases with graph database relationships to enhance knowledge retrieval.

## Overview

`go-light-rag` is a Go library that implements the core components of the LightRAG architecture rather than providing an end-to-end system. The library centers around two essential functions:

- `Insert`: Add documents to the knowledge base with flexible processing options
- `Query`: Retrieve contextually relevant information while preserving raw results

Unlike the original Python implementation which offers a complete RAG solution, this library deliberately separates the document processing pipeline from prompt engineering concerns. This approach gives developers:

1. Full control over document insertion workflows
2. Direct access to retrieved context data
3. Freedom to craft custom prompts tailored to specific use cases
4. Ability to integrate with existing Go applications and workflows

The minimalist API combined with powerful extension points makes `go-light-rag` ideal for developers who need the benefits of hybrid retrieval without being constrained by predefined prompt templates or processing pipelines.

## Architecture

`go-light-rag` is built on well-defined interfaces that enable flexibility, extensibility, and modular design. These interfaces define the contract between components, allowing you to replace or extend functionality without modifying the core logic.

### 1. Language Models (LLM)

The LLM interface abstracts different language model providers with these implementations included:

- **OpenAI**: Full support for GPT models
- **Anthropic**: Integration with Claude models
- **Ollama**: Self-hosted option for open-source models
- **OpenRouter**: Unified access to multiple model providers

Custom implementations can be created by implementing the LLM interface, which requires only a `Chat()` method.

### 2. Storage

The library defines three storage interfaces:

- **GraphStorage**: Manages entity and relationship data
- **VectorStorage**: Provides semantic search capabilities
- **KeyValueStorage**: Stores original document chunks

#### Implementations Provided

- GraphStorage: [Neo4j](https://github.com/neo4j/neo4j-go-driver) (and any compatible graph database)
- VectorStorage: [ChromeM](https://github.com/philippgille/chromem-go), [Milvus](https://github.com/milvus-io/milvus)
- KeyValueStorage: [BoltDB](https://github.com/etcd-io/bbolt), [Redis](https://github.com/redis/go-redis)

You can implement any of these interfaces to use different storage solutions.

### 3. Handlers

Handlers control document and query processing:

- **DocumentHandler**: Controls chunking, entity extraction, and processing
- **QueryHandler**: Manages keyword extraction and prompt structuring

#### Included Handlers

- **Default**: General-purpose text document processing that follows the official Python implementation. Using the zero-value for Default handler will use the same configuration as the Python implementation.
- **Semantic**: Advanced handler that extends Default to create semantically meaningful chunks by leveraging LLM to identify natural content boundaries rather than fixed token counts. Improves RAG quality at the cost of additional LLM calls.
- **Go**: Specialized handler for Go source code using AST parsing to divide code into logical sections like functions, types, and declarations.

Custom handlers can embed existing handlers and override only specific methods.

## Usage Examples

### Document Insertion

```go
// Initialize LLM
llm := llm.NewOpenAI(apiKey, model, params, logger)

// Initialize storage components
graphDB, _ := storage.NewNeo4J("bolt://localhost:7687", "neo4j", "password")

embeddingFunc := storage.EmbeddingFunc(chromem.NewEmbeddingFuncOpenAI("open_ai_key", chromem.EmbeddingModelOpenAI3Large))

// Option 1: Use ChromeM for vector storage
vecDB, _ := storage.NewChromem("vec.db", 5, embeddingFunc)

// Option 2: Use Milvus for vector storage instead
// vectorDim := 1536 // Dimension for embeddings (e.g. 1536 for OpenAI)
// milvusCfg := &milvusclient.ClientConfig{
// 	Address:  os.Getenv("MILVUS_ADDRESS"),
// 	Username: os.Getenv("MILVUS_USER"),
// 	Password: os.Getenv("MILVUS_PASSWORD"),
// 	DBName:   os.Getenv("MILVUS_DB"),
// }
// vecDB, _ := storage.NewMilvus(milvusCfg, 5, vectorDim, embeddingFunc)

// Use BoltDB for key-value storage
kvDB, _ := storage.NewBolt("kv.db")
// Or use Redis instead
// kvDB, _ := storage.NewRedis("localhost:6379", "", 0)

store := storageWrapper{
    Bolt:    kvDB,
    // Redis:   kvDB, // If redis is used, use kvDB from Option 2
    Chromem: vecDB,
    // Milvus:  vecDB, // If milvus is used, use vecDB from Option 2
    Neo4J:   graphDB,
}

// Use default document handler with zero values to match Python implementation behavior
handler := handler.Default{}

// Insert a document
doc := golightrag.Document{
    ID:      "unique-document-id",
    Content: documentContent,
}

// Chunk the document first
sources, err := golightrag.ChunkDocument(doc, handler, logger)
if err != nil {
    return err
}

// Then insert the chunked sources
err = golightrag.Insert(sources, handler, store, llm, logger)
```

### Query Processing

```go
// Create a conversation with the user's query
conversation := []golightrag.QueryConversation{
    {
        Role:    golightrag.RoleUser,
        Message: "What do you know about the main characters?",
    },
}

// Execute the query
result, err := golightrag.Query(conversation, handler, store, llm, logger)
if err != nil {
    log.Fatalf("Error processing query: %v", err)
}

// Access the retrieved context
fmt.Printf("Found %d local entities and %d global entities\n", 
    len(result.LocalEntities), len(result.GlobalEntities))

// Process source documents
for _, source := range result.LocalSources {
    fmt.Printf("Source ID: %s\nRelevance: %.2f\nContent: %s\n\n", 
        source.ID, source.Relevance, source.Content)
}

// Or use the convenient String method for formatted results
fmt.Println(result)
```

## Handler Configuration Tips

1. **Choose the right handler for your documents**:
   - `Default` for general text
   - `Semantic` for improved content comprehension where cost is less important
   - `Go` for Go source code
   - Create custom handlers for specialized content

2. **Optimize chunking parameters**:
   - Larger chunks provide more context but may exceed token limits
   - Smaller chunks process faster but may lose context
   - Balance overlap to maintain concept continuity
   - Consider `Semantic` handler for content where natural boundaries are important

3. **When using the Semantic handler**:
   - Set appropriate `TokenThreshold` based on your LLM context window
   - Configure `MaxChunkSize` to limit individual chunk sizes
   - Provide a reliable LLM instance as it's required for semantic analysis

4. **Configure concurrency appropriately**:
   - Higher concurrency speeds up processing but increases resource usage
   - Balance according to your hardware capabilities and LLM rate limits

5. **Customize entity types**:
   - Define entity types relevant to your domain
   - Be specific enough to capture important concepts
   - Be general enough to avoid excessive fragmentation

## Benchmarks

`go-light-rag` includes benchmark tests comparing its performance against a NaiveRAG implementation. The benchmarks use the same evaluation prompts as the Python implementation but with different documents and queries.

For detailed benchmark results and methodology, visit the [benchmark directory](tests/).

## Examples and Documentation

For more detailed examples, please refer to the [examples directory](examples/).
