package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/philippgille/chromem-go"
)

// Chromem provides a vector storage implementation using ChromeM database.
// It handles operations for storing and retrieving vector-based entities and relationships
// with semantic search capabilities.
type Chromem struct {
	EntitiesColl      *chromem.Collection
	RelationshipsColl *chromem.Collection

	topK int
}

// NewChromem creates a new ChromeM client with the provided parameters.
// It returns an initialized Chromem struct and any error encountered during setup.
// The dbPath parameter specifies where to persist the database, topK defines the number of
// results to return in queries, and embeddingFunc provides the vector embedding capability.
func NewChromem(dbPath string, topK int, embeddingFunc chromem.EmbeddingFunc) (Chromem, error) {
	db, err := chromem.NewPersistentDB(dbPath, false)
	if err != nil {
		return Chromem{}, fmt.Errorf("failed to create chromem db: %w", err)
	}

	entitiesColl, err := db.GetOrCreateCollection("entities", nil, embeddingFunc)
	if err != nil {
		return Chromem{}, fmt.Errorf("failed to create entities collection: %w", err)
	}
	relationshipsColl, err := db.GetOrCreateCollection("relationships", nil, embeddingFunc)
	if err != nil {
		return Chromem{}, fmt.Errorf("failed to create relationships collection: %w", err)
	}

	return Chromem{
		EntitiesColl:      entitiesColl,
		RelationshipsColl: relationshipsColl,
		topK:              topK,
	}, nil
}

// VectorQueryEntity performs a semantic search for entities based on the provided keywords.
// It returns a slice of matching entity names and any error encountered during the operation.
func (c Chromem) VectorQueryEntity(keywords string) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vecRes, err := c.EntitiesColl.Query(ctx, keywords, c.topK, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query entities: %w", err)
	}

	res := make([]string, len(vecRes))
	for i, vec := range vecRes {
		entityName, ok := vec.Metadata["entity_name"]
		if !ok {
			return nil, fmt.Errorf("entity name not found in metadata")
		}
		res[i] = entityName
	}

	return res, nil
}

// VectorQueryRelationship performs a semantic search for relationships based on the provided keywords.
// It returns a slice of source-target entity pairs and any error encountered during the operation.
func (c Chromem) VectorQueryRelationship(keywords string) ([][2]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vecRes, err := c.RelationshipsColl.Query(ctx, keywords, c.topK, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to query relationships: %w", err)
	}

	res := make([][2]string, len(vecRes))
	for i, vec := range vecRes {
		sourceEntity, ok := vec.Metadata["source_entity"]
		if !ok {
			return nil, fmt.Errorf("source entity not found in metadata")
		}
		targetEntity, ok := vec.Metadata["target_entity"]
		if !ok {
			return nil, fmt.Errorf("target entity not found in metadata")
		}
		res[i] = [2]string{sourceEntity, targetEntity}
	}

	return res, nil
}

// VectorUpsertEntity creates or updates an entity with vector embedding based on its content.
// It returns an error if the database operation fails.
func (c Chromem) VectorUpsertEntity(name, content string) error {
	doc := chromem.Document{
		ID:      uuid.New().String(),
		Content: content,
		Metadata: map[string]string{
			"entity_name": name,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	return c.EntitiesColl.AddDocument(ctx, doc)
}

// VectorUpsertRelationship creates or updates a relationship with vector embedding based on its content.
// It returns an error if the database operation fails.
func (c Chromem) VectorUpsertRelationship(source, target, content string) error {
	doc := chromem.Document{
		ID:      uuid.New().String(),
		Content: content,
		Metadata: map[string]string{
			"source_entity": source,
			"target_entity": target,
		},
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	return c.RelationshipsColl.AddDocument(ctx, doc)
}
