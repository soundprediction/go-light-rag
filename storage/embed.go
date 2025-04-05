package storage

import "context"

// EmbeddingFunc is a function type for embedding text into a vector.
type EmbeddingFunc func(ctx context.Context, text string) ([]float32, error)
