package storage

import "context"

type EmbeddingFunc func(ctx context.Context, text string) ([]float32, error)
