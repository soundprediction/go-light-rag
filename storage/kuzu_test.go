package storage

import (
	"os"
	"testing"
	"time"

	golightrag "github.com/MegaGrindStone/go-light-rag"
	"github.com/kuzudb/go-kuzu"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var (
	entity1 = golightrag.GraphEntity{
		Name:         "Entity One",
		Type:         "TestObject",
		Descriptions: "This is the first entity.",
		SourceIDs:    "source1",
		CreatedAt:    time.Now().UTC().Truncate(time.Millisecond),
	}
	entity2 = golightrag.GraphEntity{
		Name:         "Entity Two",
		Type:         "TestObject",
		Descriptions: "This is the second entity.",
		SourceIDs:    "source2",
		CreatedAt:    time.Now().UTC().Truncate(time.Millisecond),
	}
	entity3 = golightrag.GraphEntity{
		Name:         "Entity Three",
		Type:         "AnotherObject",
		Descriptions: "This is the third entity.",
		SourceIDs:    "source3",
		CreatedAt:    time.Now().UTC().Truncate(time.Millisecond),
	}
	relationship12 = golightrag.GraphRelationship{
		SourceEntity: entity1.Name,
		TargetEntity: entity2.Name,
		Weight:       0.8,
		Descriptions: "Entity One is related to Entity Two",
		Keywords:     []string{"test", "relation"},
		SourceIDs:    "relSource1",
		CreatedAt:    time.Now().UTC().Truncate(time.Millisecond),
	}
	relationship23 = golightrag.GraphRelationship{
		SourceEntity: entity2.Name,
		TargetEntity: entity3.Name,
		Weight:       0.5,
		Descriptions: "Entity Two is related to Entity Three",
		Keywords:     []string{"another", "relation"},
		SourceIDs:    "relSource2",
		CreatedAt:    time.Now().UTC().Truncate(time.Millisecond),
	}
)

// setupKuzuTestDB creates a temporary KuzuDB instance for testing.
func setupKuzuTestDB(t *testing.T) *Kuzu {
	t.Helper()
	dbPath, err := os.MkdirTemp("", "kuzu-test-*")
	require.NoError(t, err)

	// Use a cleanup function to remove the database directory after the test.
	t.Cleanup(func() {
		os.RemoveAll(dbPath)
	})
	systemConfig := kuzu.DefaultSystemConfig()
	// Use default system config for tests
	k, err := NewKuzu(dbPath, systemConfig)
	require.NoError(t, err)

	return k
}

func TestNewKuzu(t *testing.T) {
	t.Run("Successful creation", func(t *testing.T) {
		k := setupKuzuTestDB(t)
		assert.NotNil(t, k.DB)
		assert.NotNil(t, k.Conn)
		k.Close()
	})

}

func TestKuzuGraphOperations(t *testing.T) {
	k := setupKuzuTestDB(t)
	defer k.Close()

	// Upsert entities first
	err := k.GraphUpsertEntity(entity1)
	require.NoError(t, err)
	err = k.GraphUpsertEntity(entity2)
	require.NoError(t, err)
	err = k.GraphUpsertEntity(entity3)
	require.NoError(t, err)

	// Upsert relationships
	err = k.GraphUpsertRelationship(relationship12)
	require.NoError(t, err)
	err = k.GraphUpsertRelationship(relationship23)
	require.NoError(t, err)

	t.Run("Get single entity", func(t *testing.T) {
		retrieved, err := k.GraphEntity(entity1.Name)
		require.NoError(t, err)
		assert.Equal(t, entity1.Name, retrieved.Name)
		assert.Equal(t, entity1.Type, retrieved.Type)
		assert.Equal(t, entity1.Descriptions, retrieved.Descriptions)
		assert.Equal(t, entity1.SourceIDs, retrieved.SourceIDs)
		assert.WithinDuration(t, entity1.CreatedAt, retrieved.CreatedAt, time.Second)
	})

	t.Run("Get non-existent entity", func(t *testing.T) {
		_, err := k.GraphEntity("non-existent-entity")
		assert.ErrorIs(t, err, golightrag.ErrEntityNotFound)
	})

	t.Run("Get single relationship", func(t *testing.T) {
		retrieved, err := k.GraphRelationship(relationship12.SourceEntity, relationship12.TargetEntity)
		require.NoError(t, err)
		assert.Equal(t, relationship12.SourceEntity, retrieved.SourceEntity)
		assert.Equal(t, relationship12.TargetEntity, retrieved.TargetEntity)
		assert.InDelta(t, relationship12.Weight, retrieved.Weight, 0.001)
		assert.Equal(t, relationship12.Descriptions, retrieved.Descriptions)
		assert.Equal(t, relationship12.Keywords, retrieved.Keywords)
		assert.WithinDuration(t, relationship12.CreatedAt, retrieved.CreatedAt, time.Second)
	})

	t.Run("Get non-existent relationship", func(t *testing.T) {
		_, err := k.GraphRelationship(entity1.Name, entity3.Name)
		assert.ErrorIs(t, err, golightrag.ErrRelationshipNotFound)
	})

	t.Run("Get multiple entities", func(t *testing.T) {
		names := []string{entity1.Name, entity3.Name, "non-existent"}
		retrievedMap, err := k.GraphEntities(names)
		require.NoError(t, err)
		require.Len(t, retrievedMap, 2)

		// Check entity 1
		retrieved1, ok := retrievedMap[entity1.Name]
		require.True(t, ok)
		assert.Equal(t, entity1.Name, retrieved1.Name)
		assert.WithinDuration(t, entity1.CreatedAt, retrieved1.CreatedAt, time.Second)

		// Check entity 3
		retrieved3, ok := retrievedMap[entity3.Name]
		require.True(t, ok)
		assert.Equal(t, entity3.Name, retrieved3.Name)
		assert.WithinDuration(t, entity3.CreatedAt, retrieved3.CreatedAt, time.Second)
	})

	t.Run("Get multiple relationships", func(t *testing.T) {
		pairs := [][2]string{
			{relationship12.SourceEntity, relationship12.TargetEntity},
			{entity1.Name, entity3.Name}, // non-existent
		}
		retrievedMap, err := k.GraphRelationships(pairs)
		require.NoError(t, err)
		require.Len(t, retrievedMap, 1)

		key := relationship12.SourceEntity + "-" + relationship12.TargetEntity
		retrieved12, ok := retrievedMap[key]
		require.True(t, ok)
		assert.Equal(t, relationship12.SourceEntity, retrieved12.SourceEntity)
		assert.Equal(t, relationship12.TargetEntity, retrieved12.TargetEntity)
		assert.Equal(t, relationship12.Keywords, retrieved12.Keywords)
	})

	t.Run("Count entity relationships", func(t *testing.T) {
		names := []string{entity1.Name, entity2.Name, entity3.Name}
		counts, err := k.GraphCountEntitiesRelationships(names)
		require.NoError(t, err)
		require.Len(t, counts, 3)
		assert.Equal(t, 1, counts[entity1.Name]) // 1 outgoing
		assert.Equal(t, 2, counts[entity2.Name]) // 1 incoming, 1 outgoing
		assert.Equal(t, 1, counts[entity3.Name]) // 1 incoming
	})

	t.Run("Get related entities", func(t *testing.T) {
		names := []string{entity1.Name, entity2.Name}
		relatedMap, err := k.GraphRelatedEntities(names)
		require.NoError(t, err)
		require.Len(t, relatedMap, 2)

		// Check related to entity1
		relatedTo1, ok := relatedMap[entity1.Name]
		require.True(t, ok)
		require.Len(t, relatedTo1, 1)
		assert.Equal(t, entity2.Name, relatedTo1[0].Name)

		// Check related to entity2
		relatedTo2, ok := relatedMap[entity2.Name]
		require.True(t, ok)
		require.Len(t, relatedTo2, 2)
		// Order is not guaranteed
		relatedNames := []string{relatedTo2[0].Name, relatedTo2[1].Name}
		assert.Contains(t, relatedNames, entity1.Name)
		assert.Contains(t, relatedNames, entity3.Name)
	})

	t.Run("Upsert should update existing entity", func(t *testing.T) {
		updatedEntity1 := entity1
		updatedEntity1.Descriptions = "An updated description."
		err := k.GraphUpsertEntity(updatedEntity1)
		require.NoError(t, err)

		retrieved, err := k.GraphEntity(entity1.Name)
		require.NoError(t, err)
		assert.Equal(t, "An updated description.", retrieved.Descriptions)
		assert.Equal(t, entity1.Type, retrieved.Type) // Ensure other fields are unchanged
	})
}

func TestKuzu_Close(t *testing.T) {
	k := setupKuzuTestDB(t)
	// The setup function already creates a valid kuzu instance.
	// We just need to call Close and ensure no panics.
	k.Close()

	// Trying to use a closed connection should fail.
	_, err := k.GraphEntity("test")
	assert.Error(t, err, "Querying on a closed connection should return an error")
	assert.Contains(t, err.Error(), "Connection is closed")

	// Double close should not panic
	assert.NotPanics(t, func() {
		k.Close()
	})
}
