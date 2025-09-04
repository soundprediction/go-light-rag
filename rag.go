package golightrag

import (
	"errors"
	"fmt"
	"slices"
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
	// GraphEntity retrieves a single entity by name from the graph storage.
	// Returns ErrEntityNotFound if the entity doesn't exist.
	GraphEntity(name string) (GraphEntity, error)
	// GraphRelationship retrieves a relationship between sourceEntity and targetEntity.
	// Returns ErrRelationshipNotFound if the relationship doesn't exist.
	GraphRelationship(sourceEntity, targetEntity string) (GraphRelationship, error)

	// GraphUpsertEntity creates a new entity or updates an existing entity in the graph storage.
	// If the entity already exists, it should merge the new data with existing data.
	GraphUpsertEntity(entity GraphEntity) error
	// GraphUpsertRelationship creates a new relationship or updates an existing relationship
	// between two entities in the graph storage.
	// If the relationship already exists, it should merge the new data with existing data.
	GraphUpsertRelationship(relationship GraphRelationship) error

	// GraphEntities batch retrieves multiple entities by their names.
	// Returns a map with entity names as keys and entity objects as values.
	// If an entity doesn't exist, it should be omitted from the result map.
	GraphEntities(names []string) (map[string]GraphEntity, error)
	// GraphRelationships batch retrieves multiple relationships by their source-target pairs.
	// Returns a map with composite keys (formatted as "source-target") as keys and
	// relationship objects as values.
	// If a relationship doesn't exist, it should be omitted from the result map.
	GraphRelationships(pairs [][2]string) (map[string]GraphRelationship, error)

	// GraphCountEntitiesRelationships counts the number of relationships each entity has.
	// Returns a map with entity names as keys and relationship counts as values.
	// This is used to determine entity importance during queries.
	GraphCountEntitiesRelationships(names []string) (map[string]int, error)
	// GraphRelatedEntities finds entities directly connected to the specified entities.
	// Returns a map with entity names as keys and slices of directly connected entities as values.
	// Used to expand the context during queries.
	GraphRelatedEntities(names []string) (map[string][]GraphEntity, error)
}

// VectorStorage defines the interface for vector database operations.
// It provides methods to query and store entities and relationships
// in a vector space for semantic search capabilities.
type VectorStorage interface {
	// VectorQueryEntity performs a semantic search for entities based on the provided keywords.
	// Returns a slice of entity names that semantically match the keywords.
	// The results should be ordered by relevance.
	VectorQueryEntity(keywords string) ([]string, error)
	// VectorQueryRelationship performs a semantic search for relationships based on the provided keywords.
	// Returns a slice of source-target entity name pairs that semantically match the keywords.
	// The results should be ordered by relevance.
	VectorQueryRelationship(keywords string) ([][2]string, error)

	// VectorUpsertEntity creates or updates the vector representation of an entity.
	// The content parameter should contain the text used for semantic matching.
	// This typically includes the entity name and description.
	VectorUpsertEntity(name, content string) error
	// VectorUpsertRelationship creates or updates the vector representation of a relationship.
	// The content parameter should contain the text used for semantic matching.
	// This typically includes keywords, descriptions, and entity names.
	VectorUpsertRelationship(source, target, content string) error
}

// KeyValueStorage defines the interface for key-value storage operations.
// It provides methods to access and store source documents.
type KeyValueStorage interface {
	// KVSource retrieves a source document chunk by its ID.
	// Returns an error if the source doesn't exist or can't be retrieved.
	KVSource(id string) (Source, error)
	KVUnprocessed(id string) (string, error)
	// KVUpsertSources creates or updates multiple source document chunks at once.
	// Each source should be stored with its ID as the key.
	// This is called during document processing to store chunked documents.
	KVUpsertSources(sources []Source) error
	KVUpsertUnprocessed(sources []Source) error
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
	Name         string `json:"entity_name"`
	Type         string `json:"entity_type"`
	Descriptions string `json:"entity_description"`
	SourceIDs    string
	CreatedAt    time.Time
}

// GraphRelationship represents a relationship between two entities in the knowledge graph.
// It contains information about the source and target entities,
// relationship weight, descriptions, keywords, sources, and creation timestamp.
type GraphRelationship struct {
	SourceEntity string   `json:"source_entity"`
	TargetEntity string   `json:"target_entity"`
	Weight       float64  `json:"relationship_strength"`
	Descriptions string   `json:"relationship_description"`
	Keywords     []string `json:"relationship_keywords"`
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
	if slices.Contains(slice, item) {
		return slice
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
