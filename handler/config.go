package handler

// DocumentConfig contains configuration parameters for document processing
// during RAG operations, including retry behavior and token length limits.
type DocumentConfig struct {
	MaxRetries              int
	GleanCount              int
	MaxSummariesTokenLength int
}

const (
	defaultMaxSummariesTokenLength = 1200
)
