package golightrag_test

import (
	"encoding/json"
	"errors"
	"log/slog"
	"os"
	"testing"

	golightrag "github.com/MegaGrindStone/go-light-rag"
)

func TestQuery(t *testing.T) {
	// logger := slog.New(slog.NewTextHandler(io.Discard, nil))
	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))

	t.Run("Successful query", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Entity1",
			},
		}

		// Create mock LLM that returns valid keyword extraction
		keywordExtraction := map[string][]string{
			"high_level_keywords": {"Entity1", "Knowledge"},
			"low_level_keywords":  {"Entity1", "Information"},
		}
		keywordExtractionJSON, _ := json.Marshal(keywordExtraction)

		mockLLM := &MockLLM{
			chatResponse: string(keywordExtractionJSON),
			maxRetries:   3,
			gleanCount:   2,
			maxTokenLen:  1000,
			chatCalls:    make([][]string, 0),
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
			llm: mockLLM,
		}

		// Create mock storage with predefined data
		storage := &MockStorage{
			// Entity data
			entities: map[string]golightrag.GraphEntity{
				"ENTITY1": {
					Name:         "ENTITY1",
					Type:         "PERSON",
					Descriptions: "Description of Entity1",
					SourceIDs:    "doc-1-chunk-0",
				},
				"ENTITY2": {
					Name:         "ENTITY2",
					Type:         "ORGANIZATION",
					Descriptions: "Description of Entity2",
					SourceIDs:    "doc-1-chunk-0",
				},
			},
			// Relationship data
			relationships: map[string]golightrag.GraphRelationship{
				"ENTITY1:ENTITY2": {
					SourceEntity: "ENTITY1",
					TargetEntity: "ENTITY2",
					Descriptions: "Entity1 is related to Entity2",
					Keywords:     "RELATED_TO",
					Weight:       1.0,
					SourceIDs:    "doc-1-chunk-0",
				},
			},
			// Vector query results
			vectorQueryEntityResults: []string{"ENTITY1"},
			vectorQueryRelationshipResults: [][2]string{
				{"ENTITY1", "ENTITY2"},
			},
			// For related entities lookup
			entityRelatedEntitiesMap: map[string][]golightrag.GraphEntity{
				"ENTITY1": {
					{
						Name:         "ENTITY2",
						Type:         "ORGANIZATION",
						Descriptions: "Description of Entity2",
						SourceIDs:    "doc-1-chunk-0",
					},
				},
			},
			// For entity relationship counts
			entityRelationshipCountMap: map[string]int{
				"ENTITY1": 1,
				"ENTITY2": 1,
			},
			// Source data
			sources: map[string]golightrag.Source{
				"doc-1-chunk-0": {
					ID:         "doc-1-chunk-0",
					Content:    "Content about Entity1 and Entity2",
					TokenSize:  10,
					OrderIndex: 0,
				},
			},
		}

		// Call the function under test
		result, err := golightrag.Query(conversations, handler, storage, logger)
		// Assertions
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		// Check that the result contains entities and relationships
		if len(result.LocalEntities) == 0 && len(result.GlobalEntities) == 0 {
			t.Error("Expected entities in result, got none")
		}

		if len(result.LocalRelationships) == 0 && len(result.GlobalRelationships) == 0 {
			t.Error("Expected relationships in result, got none")
		}

		// Check chat was called with expected prompt
		if len(mockLLM.chatCalls) == 0 {
			t.Error("Expected LLM to be called")
		}
	})

	t.Run("Error in extracting query", func(t *testing.T) {
		// Create conversations with no user message
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleAssistant,
				Message: "I am an assistant",
			},
		}

		// Create mock handler and storage
		handler := &MockQueryHandler{}
		storage := &MockStorage{}

		// Call the function under test
		_, err := golightrag.Query(conversations, handler, storage, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to missing user message, got nil")
		}
	})

	t.Run("Error in LLM chat", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Entity1",
			},
		}

		// Create mock LLM with error
		mockLLM := &MockLLM{
			chatErr:     errors.New("LLM chat error"),
			maxRetries:  3,
			gleanCount:  2,
			maxTokenLen: 1000,
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
			llm: mockLLM,
		}

		// Create mock storage
		storage := &MockStorage{}

		// Call the function under test
		_, err := golightrag.Query(conversations, handler, storage, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to LLM chat error, got nil")
		}
	})

	t.Run("Error in JSON unmarshaling", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Entity1",
			},
		}

		// Create mock LLM with invalid JSON
		mockLLM := &MockLLM{
			chatResponse: "this is not valid JSON",
			maxRetries:   3,
			gleanCount:   2,
			maxTokenLen:  1000,
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
			llm: mockLLM,
		}

		// Create mock storage
		storage := &MockStorage{}

		// Call the function under test
		_, err := golightrag.Query(conversations, handler, storage, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to invalid JSON, got nil")
		}
	})

	t.Run("Error in vector query entity", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Entity1",
			},
		}

		// Create mock LLM with valid response
		keywordExtraction := map[string][]string{
			"high_level_keywords": {"Entity1"},
			"low_level_keywords":  {"Entity1"},
		}
		keywordExtractionJSON, _ := json.Marshal(keywordExtraction)

		mockLLM := &MockLLM{
			chatResponse: string(keywordExtractionJSON),
			maxRetries:   3,
			gleanCount:   2,
			maxTokenLen:  1000,
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
			llm: mockLLM,
		}

		// Create mock storage with vector query error
		storage := &MockStorage{
			vectorQueryEntityErr: errors.New("vector query entity error"),
		}

		// Call the function under test
		_, err := golightrag.Query(conversations, handler, storage, logger)

		// Assertions
		if err == nil {
			t.Error("Expected error due to vector query entity error, got nil")
		}
	})

	t.Run("Empty query results", func(t *testing.T) {
		// Mock conversations
		conversations := []golightrag.QueryConversation{
			{
				Role:    golightrag.RoleUser,
				Message: "Tell me about Unknown",
			},
		}

		// Create mock LLM with valid response
		keywordExtraction := map[string][]string{
			"high_level_keywords": {"Unknown"},
			"low_level_keywords":  {"Unknown"},
		}
		keywordExtractionJSON, _ := json.Marshal(keywordExtraction)

		mockLLM := &MockLLM{
			chatResponse: string(keywordExtractionJSON),
			maxRetries:   3,
			gleanCount:   2,
			maxTokenLen:  1000,
		}

		// Create mock handler
		handler := &MockQueryHandler{
			keywordExtractionPromptData: golightrag.KeywordExtractionPromptData{
				Goal: "Extract keywords",
			},
			llm: mockLLM,
		}

		// Create mock storage with empty results
		storage := &MockStorage{
			vectorQueryEntityResults:       []string{},
			vectorQueryRelationshipResults: [][2]string{},
		}

		// Call the function under test
		result, err := golightrag.Query(conversations, handler, storage, logger)
		// Assertions
		if err != nil {
			t.Errorf("Expected no error, got %v", err)
		}

		// Check that results are empty
		if len(result.LocalEntities) != 0 {
			t.Errorf("Expected 0 local entities, got %d", len(result.LocalEntities))
		}
		if len(result.GlobalEntities) != 0 {
			t.Errorf("Expected 0 global entities, got %d", len(result.GlobalEntities))
		}
	})
}
