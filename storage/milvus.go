package storage

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"
	"github.com/milvus-io/milvus/client/v2/entity"
	"github.com/milvus-io/milvus/client/v2/index"
	"github.com/milvus-io/milvus/client/v2/milvusclient"
)

// Milvus provides a vector storage implementation using Milvus database.
// It handles operations for storing and retrieving vector-based entities and relationships
// with semantic search capabilities.
//
// The Close() method should be called when done to properly release resources.
type Milvus struct {
	client        *milvusclient.Client
	embeddingFunc EmbeddingFunc
	vectorDim     int
	topK          int
}

const (
	milvusEntitiesCollectionName      = "entities"
	milvusRelationshipsCollectionName = "relationships"

	cosineThreshold = 0.2
)

// NewMilvus creates a new Milvus client with the provided parameters.
// It returns an initialized Milvus struct and any error encountered during setup.
func NewMilvus(config *milvusclient.ClientConfig, topK, vectorDim int, embeddingFunc EmbeddingFunc) (Milvus, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	// Connect to Milvus
	c, err := milvusclient.New(ctx, config)
	if err != nil {
		return Milvus{}, fmt.Errorf("failed to connect to Milvus: %w", err)
	}

	m := Milvus{
		client:        c,
		embeddingFunc: embeddingFunc,
		vectorDim:     vectorDim,
		topK:          topK,
	}

	if err := m.createEntitiesCollection(ctx); err != nil {
		return Milvus{}, err
	}

	if err := m.createRelationshipsCollection(ctx); err != nil {
		return Milvus{}, err
	}

	return m, nil
}

// VectorQueryEntity performs a semantic search for entities based on the provided keywords.
func (m Milvus) VectorQueryEntity(keywords string) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vector, err := m.embeddingFunc(ctx, keywords)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for query: %w", err)
	}
	vectors := []entity.Vector{entity.FloatVector(vector)}

	annParam := index.NewCustomAnnParam()
	annParam.WithRadius(cosineThreshold)
	opt := milvusclient.
		NewSearchOption(milvusEntitiesCollectionName, m.topK, vectors).
		WithAnnParam(annParam)
	searchResult, err := m.client.Search(ctx, opt)
	if err != nil {
		return nil, fmt.Errorf("failed to query entities: %w", err)
	}

	results := make([]string, 0, m.topK)
	for _, result := range searchResult {
		for i := 0; i < result.ResultCount; i++ {
			entityName, err := result.GetColumn("entity_name").Get(i)
			if err != nil {
				return nil, fmt.Errorf("failed to get entity name from result: %w", err)
			}
			entityNameStr, ok := entityName.(string)
			if !ok {
				return nil, fmt.Errorf("entity name not string")
			}
			results = append(results, entityNameStr)
		}
	}

	return results, nil
}

// VectorQueryRelationship performs a semantic search for relationships based on the provided keywords.
func (m Milvus) VectorQueryRelationship(keywords string) ([][2]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	vector, err := m.embeddingFunc(ctx, keywords)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for query: %w", err)
	}
	vectors := []entity.Vector{entity.FloatVector(vector)}

	annParam := index.NewCustomAnnParam()
	annParam.WithRadius(cosineThreshold)
	opt := milvusclient.
		NewSearchOption(milvusRelationshipsCollectionName, m.topK, vectors).
		WithAnnParam(annParam)
	searchResult, err := m.client.Search(ctx, opt)
	if err != nil {
		return nil, fmt.Errorf("failed to query relationships: %w", err)
	}

	results := make([][2]string, 0, m.topK)
	for _, result := range searchResult {
		for i := 0; i < result.ResultCount; i++ {
			sourceEntity, err := result.GetColumn("source_entity").Get(i)
			if err != nil {
				return nil, fmt.Errorf("failed to get source entity from result: %w", err)
			}
			sourceEntityStr, ok := sourceEntity.(string)
			if !ok {
				return nil, fmt.Errorf("source entity not string")
			}

			targetEntity, err := result.GetColumn("target_entity").Get(i)
			if err != nil {
				return nil, fmt.Errorf("failed to get target entity from result: %w", err)
			}
			targetEntityStr, ok := targetEntity.(string)
			if !ok {
				return nil, fmt.Errorf("target entity not string")
			}

			results = append(results, [2]string{sourceEntityStr, targetEntityStr})
		}
	}

	return results, nil
}

// VectorUpsertEntity creates or updates an entity with vector embedding based on its content.
func (m Milvus) VectorUpsertEntity(name, content string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	vector, err := m.embeddingFunc(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding for entity: %w", err)
	}

	opt := milvusclient.NewColumnBasedInsertOption(milvusEntitiesCollectionName).
		WithVarcharColumn("id", []string{uuid.New().String()}).
		WithVarcharColumn("entity_name", []string{name}).
		WithFloatVectorColumn("vector", m.vectorDim, [][]float32{vector})
	_, err = m.client.Upsert(ctx, opt)
	if err != nil {
		return fmt.Errorf("failed to upsert entity: %w", err)
	}

	return nil
}

// VectorUpsertRelationship creates or updates a relationship with vector embedding based on its content.
func (m Milvus) VectorUpsertRelationship(source, target, content string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	vector, err := m.embeddingFunc(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding for relationship: %w", err)
	}

	opt := milvusclient.NewColumnBasedInsertOption(milvusRelationshipsCollectionName).
		WithVarcharColumn("id", []string{uuid.New().String()}).
		WithVarcharColumn("source_entity", []string{source}).
		WithVarcharColumn("target_entity", []string{target}).
		WithFloatVectorColumn("vector", m.vectorDim, [][]float32{vector})
	_, err = m.client.Upsert(ctx, opt)
	if err != nil {
		return fmt.Errorf("failed to upsert relationship: %w", err)
	}

	return nil
}

// Close closes the connection to Milvus.
func (m Milvus) Close(ctx context.Context) error {
	if m.client != nil {
		return m.client.Close(ctx)
	}
	return nil
}

func (m Milvus) createEntitiesCollection(ctx context.Context) error {
	has, err := m.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(milvusEntitiesCollectionName))
	if err != nil {
		return fmt.Errorf("failed to check if entities collection exists: %w", err)
	}

	if has {
		return nil
	}

	err = m.client.CreateCollection(ctx,
		milvusclient.SimpleCreateCollectionOptions(milvusEntitiesCollectionName, int64(m.vectorDim)).
			WithVarcharPK(true, 64))
	if err != nil {
		return fmt.Errorf("failed to create entities collection: %w", err)
	}

	return nil
}

func (m Milvus) createRelationshipsCollection(ctx context.Context) error {
	has, err := m.client.HasCollection(ctx, milvusclient.NewHasCollectionOption(milvusRelationshipsCollectionName))
	if err != nil {
		return fmt.Errorf("failed to check if relationships collection exists: %w", err)
	}

	if has {
		return nil
	}

	err = m.client.CreateCollection(ctx,
		milvusclient.SimpleCreateCollectionOptions(milvusRelationshipsCollectionName, int64(m.vectorDim)).
			WithVarcharPK(true, 64))
	if err != nil {
		return fmt.Errorf("failed to create relationships collection: %w", err)
	}

	return nil
}
