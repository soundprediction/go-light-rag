package golightrag_test

import (
	"fmt"

	golightrag "github.com/MegaGrindStone/go-light-rag"
)

type MockDocument struct {
	id      string
	content string
}

type MockDocumentProcessor struct {
	chunkErr                   error
	sources                    []golightrag.Source
	entityExtractionPromptData golightrag.EntityExtractionPromptData
	llm                        golightrag.LLM
}

type MockLLM struct {
	chatResponse string
	chatErr      error
	maxRetries   int
	gleanCount   int
	maxTokenLen  int

	// For tracking interactions
	chatCalls [][]string
}

type MockStorage struct {
	vectorUpsertSourcesErr      error
	kvUpsertSourcesErr          error
	graphEntityErr              error
	graphRelationshipErr        error
	graphUpsertEntityErr        error
	vectorUpsertEntityErr       error
	graphUpsertRelationshipErr  error
	vectorUpsertRelationshipErr error

	// Track calls to methods
	vectorUpsertSourcesCalled bool
	kvUpsertSourcesCalled     bool

	// Track entities and relationships
	entities      map[string]golightrag.GraphEntity
	relationships map[string]golightrag.GraphRelationship
}

func (m MockDocument) ID() string {
	return m.id
}

func (m MockDocument) Content() string {
	return m.content
}

func (m *MockDocumentProcessor) ChunksDocument(string) ([]golightrag.Source, error) {
	if m.chunkErr != nil {
		return nil, m.chunkErr
	}
	return m.sources, nil
}

func (m *MockDocumentProcessor) EntityExtractionPromptData() golightrag.EntityExtractionPromptData {
	return m.entityExtractionPromptData
}

func (m *MockDocumentProcessor) LLM() golightrag.LLM {
	return m.llm
}

func (m *MockLLM) Chat(messages []string) (string, error) {
	// Record this call
	if m.chatCalls != nil {
		m.chatCalls = append(m.chatCalls, messages)
	}

	if m.chatErr != nil {
		return "", m.chatErr
	}
	return m.chatResponse, nil
}

func (m *MockLLM) MaxRetries() int {
	return m.maxRetries
}

func (m *MockLLM) GleanCount() int {
	return m.gleanCount
}

func (m *MockLLM) MaxSummariesTokenLength() int {
	return m.maxTokenLen
}

func (m *MockStorage) VectorUpsertSources(string, []golightrag.Source) error {
	m.vectorUpsertSourcesCalled = true
	return m.vectorUpsertSourcesErr
}

func (m *MockStorage) KVUpsertSources(string, []golightrag.Source) error {
	m.kvUpsertSourcesCalled = true
	return m.kvUpsertSourcesErr
}

func (m *MockStorage) GraphEntity(name string) (golightrag.GraphEntity, error) {
	if m.graphEntityErr != nil {
		return golightrag.GraphEntity{}, m.graphEntityErr
	}

	// Check if the entity exists in our tracked map
	if entity, exists := m.entities[name]; exists {
		return entity, nil
	}

	return golightrag.GraphEntity{}, golightrag.ErrEntityNotFound
}

func (m *MockStorage) GraphRelationship(sourceEntity, targetEntity string) (golightrag.GraphRelationship, error) {
	if m.graphRelationshipErr != nil {
		return golightrag.GraphRelationship{}, m.graphRelationshipErr
	}

	// Check if the relationship exists in our tracked map
	key := fmt.Sprintf("%s:%s", sourceEntity, targetEntity)
	if rel, exists := m.relationships[key]; exists {
		return rel, nil
	}

	return golightrag.GraphRelationship{}, golightrag.ErrRelationshipNotFound
}

func (m *MockStorage) GraphUpsertEntity(entity golightrag.GraphEntity) error {
	if m.graphUpsertEntityErr != nil {
		return m.graphUpsertEntityErr
	}

	if m.entities == nil {
		m.entities = make(map[string]golightrag.GraphEntity)
	}

	// Store the entity
	m.entities[entity.Name] = entity
	return nil
}

func (m *MockStorage) VectorUpsertEntity(entity golightrag.GraphEntity) error {
	if m.vectorUpsertEntityErr != nil {
		return m.vectorUpsertEntityErr
	}

	if m.entities == nil {
		m.entities = make(map[string]golightrag.GraphEntity)
	}

	// For simplicity, we'll store in the same map
	m.entities[entity.Name] = entity
	return nil
}

func (m *MockStorage) GraphUpsertRelationship(rel golightrag.GraphRelationship) error {
	if m.graphUpsertRelationshipErr != nil {
		return m.graphUpsertRelationshipErr
	}

	if m.relationships == nil {
		m.relationships = make(map[string]golightrag.GraphRelationship)
	}

	// Store the relationship
	key := fmt.Sprintf("%s:%s", rel.SourceEntity, rel.TargetEntity)
	m.relationships[key] = rel
	return nil
}

func (m *MockStorage) VectorUpsertRelationship(rel golightrag.GraphRelationship) error {
	if m.vectorUpsertRelationshipErr != nil {
		return m.vectorUpsertRelationshipErr
	}

	if m.relationships == nil {
		m.relationships = make(map[string]golightrag.GraphRelationship)
	}

	// For simplicity, we'll store in the same map
	key := fmt.Sprintf("%s:%s", rel.SourceEntity, rel.TargetEntity)
	m.relationships[key] = rel
	return nil
}
