package golightrag

import (
	"errors"
	"fmt"
	"strings"
	"text/template"
	"time"
)

// LLM defines the interface for language model operations.
// It provides methods for chat interaction, handling retries,
// extracting information, and managing token limits.
type LLM interface {
	// Chat sends messages to the LLM and returns the response.
	// A message with an even index is guaranteed to be sent by the user, while the odd index is
	// sent by the assistant.
	Chat(messages []string) (string, error)
}

// GraphStorage defines the interface for graph database operations.
// It provides methods to query and manipulate entities and relationships
// in a knowledge graph.
type GraphStorage interface {
	GraphEntity(name string) (GraphEntity, error)
	GraphRelationship(sourceEntity, targetEntity string) (GraphRelationship, error)

	GraphUpsertEntity(entity GraphEntity) error
	GraphUpsertRelationship(relationship GraphRelationship) error

	GraphCountEntityRelationships(name string) (int, error)
	GraphRelatedEntities(name string) ([]GraphEntity, error)
}

// VectorStorage defines the interface for vector database operations.
// It provides methods to query and store entities and relationships
// in a vector space for semantic search capabilities.
type VectorStorage interface {
	VectorQueryEntity(keywords string) ([]string, error)
	VectorQueryRelationship(keywords string) ([][2]string, error)

	VectorUpsertEntity(name, content string) error
	VectorUpsertRelationship(source, target, content string) error
}

// KeyValueStorage defines the interface for key-value storage operations.
// It provides methods to access and store source documents.
type KeyValueStorage interface {
	KVSource(id string) (Source, error)
	KVUpsertSources(sources []Source) error
}

// Storage is a composite interface that combines GraphStorage,
// VectorStorage, and KeyValueStorage interfaces to provide
// comprehensive data storage capabilities.
type Storage interface {
	GraphStorage
	VectorStorage
	KeyValueStorage
}

// Source represents a document chunk with metadata.
// It contains the text content, size information, and position data.
type Source struct {
	ID         string
	Content    string
	TokenSize  int
	OrderIndex int
}

// GraphEntity represents an entity in the knowledge graph.
// It contains information about the entity's name, type,
// descriptions, sources, and creation timestamp.
type GraphEntity struct {
	Name         string
	Type         string
	Descriptions string
	SourceIDs    string
	CreatedAt    time.Time
}

// GraphRelationship represents a relationship between two entities in the knowledge graph.
// It contains information about the source and target entities,
// relationship weight, descriptions, keywords, sources, and creation timestamp.
type GraphRelationship struct {
	SourceEntity string
	TargetEntity string
	Weight       float64
	Descriptions string
	Keywords     string
	SourceIDs    string
	CreatedAt    time.Time
}

var (
	// ErrEntityNotFound is returned when an entity is not found in the storage.
	ErrEntityNotFound = errors.New("entity not found")
	// ErrRelationshipNotFound is returned when a relationship is not found in the storage.
	ErrRelationshipNotFound = errors.New("relationship not found")
)

func cleanContent(content string) string {
	// Removes spaces and null characters.
	str := strings.TrimSpace(content)
	return strings.ReplaceAll(str, "\x00", "")
}

func promptTemplate(name, templ string, data any) (string, error) {
	buf := strings.Builder{}
	tmpl := template.New(name).Funcs(template.FuncMap{
		"add": func(a, b int) int {
			return a + b
		},
	})
	tmpl = template.Must(tmpl.Parse(templ))
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

func appendIfUnique(slice []string, item string) []string {
	for _, ele := range slice {
		if ele == item {
			return slice
		}
	}
	return append(slice, item)
}

func mostFrequentItem(list []string) string {
	// Create a map to store counts
	counts := make(map[string]int)

	// Count occurrences of each string
	for _, item := range list {
		counts[item]++
	}

	// Find the item with highest count
	maxCount := 0
	var mostFreqItem string

	for item, count := range counts {
		if count > maxCount {
			maxCount = count
			mostFreqItem = item
		}
	}

	return mostFreqItem
}

func threeBacktick(caption string) string {
	return "```" + caption
}

func (s Source) genID(docID string) string {
	return fmt.Sprintf("%s-chunk-%d", docID, s.OrderIndex)
}
