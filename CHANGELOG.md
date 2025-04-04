# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Add `Redis` struct for Redis key-value database integration.

## [0.1.1] - 2025-04-01

This update introduces semantic text chunking capabilities through a new handler and improves code quality with comprehensive tests, while also addressing a concurrency setting issue that could impact processing performance.

### Added

- Add `Semantic` handler that intelligently chunks text documents based on content meaning rather than arbitrary size limits.
- Add comprehensive unit tests for the handler package to ensure reliability and correctness.

### Fixed

- Fix incorrect default value in `Default` handler's concurrency count setting that could affect processing performance.

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
