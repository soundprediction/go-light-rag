# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `Semantic` handler for semantic chunking of text documents.
- Add unit tests for handler package

### Fixed

- Fix `Default` concurrency count default value.

## [0.1.0] - 2025-03-25

This initial release introduces a comprehensive data access framework with support for multiple database types (graph, vector, and key-value) and AI service integrations. New functionality includes query and insertion operations, document processing capabilities for both standard text and Go source code, and standardized interfaces for future extensibility.

### Added

- Add `Query` function for database querying operations.
- Add `Insert` function for database insertion operations.
- Implement core interfaces required by `Query` and `Insert` functions.
- Add `Neo4J` struct for Neo4j graph database integration.
- Add `Chromem` struct for `chromem-go` vector database integration.
- Add `Bolt` struct for `bbolt` key-value database integration.
- Implement AI service integrations (`Anthropic`, `Ollama`, `OpenAI`, and `OpenRouter`) through the `LLM` interface.
- Add `Default` struct for processing standard text documents.
- Add `Go` struct for parsing and analyzing Go source code.
