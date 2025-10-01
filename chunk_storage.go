package golightrag

import (
	"context"
	"fmt"
)

// ChunkStorage defines the interface for storing and retrieving content chunks and embeddings.
type ChunkStorage interface {
	// InsertChunks stores multiple content chunks
	InsertChunks(ctx context.Context, chunks []ContentChunk) error

	// GetChunk retrieves a single chunk by ID
	GetChunk(ctx context.Context, chunkID string) (*ContentChunk, error)

	// GetChunksForContent retrieves all chunks for a given content ID
	GetChunksForContent(ctx context.Context, contentID string) ([]ContentChunk, error)

	// InsertEmbedding stores an embedding for a chunk
	InsertEmbedding(ctx context.Context, embedding ContentEmbedding) error

	// GetEmbeddingsForChunk retrieves all embeddings for a chunk
	GetEmbeddingsForChunk(ctx context.Context, chunkID string) ([]ContentEmbedding, error)

	// GetChunksWithEmbeddings retrieves chunks with their embeddings for a specific model
	GetChunksWithEmbeddings(ctx context.Context, model string) ([]ContentChunk, error)

	// GetUnembeddedChunks retrieves chunks that don't have embeddings for the specified model
	GetUnembeddedChunks(ctx context.Context, model string) ([]ContentChunk, error)
}

// InsertChunksWithStorage is a convenience function that stores chunks using ChunkStorage.
// This provides a simpler alternative to the InsertChunks function that uses the Storage interface.
func InsertChunksWithStorage(ctx context.Context, chunks []ContentChunk, storage ChunkStorage, logger interface{}) error {
	if len(chunks) == 0 {
		return nil
	}

	if err := storage.InsertChunks(ctx, chunks); err != nil {
		return fmt.Errorf("failed to insert chunks: %w", err)
	}

	return nil
}

// EmbedChunks generates embeddings for chunks that don't have them for the specified model.
// It retrieves unembedded chunks, generates embeddings, and stores them.
func EmbedChunks(ctx context.Context, storage ChunkStorage, embedder interface{}, model string) error {
	// Get chunks without embeddings for this model
	chunks, err := storage.GetUnembeddedChunks(ctx, model)
	if err != nil {
		return fmt.Errorf("failed to get unembedded chunks: %w", err)
	}

	if len(chunks) == 0 {
		return nil // No chunks to embed
	}

	// TODO: Generate embeddings and store them
	// This would require an embedder interface to be defined
	// For now, this is a placeholder that returns an error

	return fmt.Errorf("embedding generation not yet implemented")
}
