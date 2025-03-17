package handler

import (
	"fmt"
	"math"
	"strings"

	golightrag "github.com/MegaGrindStone/go-light-rag"
	"github.com/MegaGrindStone/go-light-rag/internal"
)

// Default implements both DocumentHandler and QueryHandler interfaces for RAG operations.
// It provides configurable handling for document chunking, entity extraction, and keyword extraction
// with sensible defaults.
type Default struct {
	ChunkMaxTokenSize     int
	ChunkOverlapTokenSize int

	EntityExtractionGoal     string
	EntityTypes              []string
	Language                 string
	EntityExtractionExamples []golightrag.EntityExtractionPromptExample

	KeywordExtractionGoal     string
	KeywordExtractionExamples []golightrag.KeywordExtractionPromptExample
}

const (
	defaultChunkMaxTokenSize     = 1200
	defaultChunkOverlapTokenSize = 100
)

// ChunksDocument splits a document's content into overlapping chunks of text.
// It uses tiktoken to encode and decode tokens, and returns an array of Source objects.
// Each Source contains a portion of the original text with appropriate metadata.
// It returns an error if encoding or decoding fails.
func (d Default) ChunksDocument(content string) ([]golightrag.Source, error) {
	tokenIDs, err := internal.EncodeStringByTiktoken(content)
	if err != nil {
		return nil, fmt.Errorf("failed to encode string: %w", err)
	}

	maxTokenSize := d.ChunkMaxTokenSize
	if maxTokenSize == 0 {
		maxTokenSize = defaultChunkMaxTokenSize
	}
	overlapTokenSize := d.ChunkOverlapTokenSize
	if overlapTokenSize == 0 {
		overlapTokenSize = defaultChunkOverlapTokenSize
	}

	results := []golightrag.Source{}
	for index, start := 0, 0; start < len(tokenIDs); index, start = index+1, start+maxTokenSize-overlapTokenSize {
		end := start + maxTokenSize
		if end > len(tokenIDs) {
			end = len(tokenIDs)
		}

		chunkContent, err := internal.DecodeTokensByTiktoken(tokenIDs[start:end])
		if err != nil {
			return nil, fmt.Errorf("failed to decode tokens: %w", err)
		}

		results = append(results, golightrag.Source{
			Content:    strings.TrimSpace(chunkContent),
			TokenSize:  int(math.Min(float64(maxTokenSize), float64(len(tokenIDs)-start))),
			OrderIndex: index,
		})
	}

	return results, nil
}

// EntityExtractionPromptData returns the data needed to generate prompts for extracting
// entities and relationships from text content.
func (d Default) EntityExtractionPromptData() golightrag.EntityExtractionPromptData {
	return golightrag.EntityExtractionPromptData{
		Goal:        d.EntityExtractionGoal,
		EntityTypes: d.EntityTypes,
		Language:    d.Language,
		Examples:    d.EntityExtractionExamples,
	}
}

// KeywordExtractionPromptData returns the data needed to generate prompts for extracting
// keywords from user queries and conversation history.
func (d Default) KeywordExtractionPromptData() golightrag.KeywordExtractionPromptData {
	return golightrag.KeywordExtractionPromptData{
		Goal:     d.KeywordExtractionGoal,
		Examples: d.KeywordExtractionExamples,
	}
}
