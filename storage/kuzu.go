package storage

import (
	"context"
	"fmt"
	"strings"
	"time"

	kuzudb "github.com/kuzudb/go-kuzu"
)

// EmbeddingFunc defines the function signature for converting a string into a vector embedding.
// It is now context-aware to handle timeouts and cancellations.

// Kuzu provides a vector storage implementation using the Kuzu graph database.
// It handles operations for storing and retrieving vector-based entities and relationships
// with semantic search capabilities.
type Kuzu struct {
	DB            *kuzudb.Database
	Conn          *kuzudb.Connection
	topK          int
	embeddingFunc EmbeddingFunc

	// Prepared statements for efficient query execution
	stmtQueryEntity        *kuzudb.PreparedStatement
	stmtQueryRelationship  *kuzudb.PreparedStatement
	stmtUpsertEntity       *kuzudb.PreparedStatement
	stmtUpsertRelationship *kuzudb.PreparedStatement
}

// NewKuzu creates a new Kuzu client, initializes the database schema, and prepares query statements.
// It returns an initialized Kuzu struct and any error encountered during setup.
func NewKuzu(dbPath string, topK int, config kuzudb.SystemConfig, embeddingFunc EmbeddingFunc) (*Kuzu, error) {
	if embeddingFunc == nil {
		return nil, fmt.Errorf("embedding function cannot be nil")
	}

	// Determine the dimension of the embeddings for the schema.
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	dummyEmbedding, err := embeddingFunc(ctx, "test")
	if err != nil {
		return nil, fmt.Errorf("failed to determine embedding dimension: %w", err)
	}
	embeddingDim := len(dummyEmbedding)
	if embeddingDim == 0 {
		return nil, fmt.Errorf("embedding function returned a zero-dimension vector")
	}

	// Initialize the database with a nil SystemConfig to use default settings.
	db, err := kuzudb.OpenDatabase(dbPath, config)
	if err != nil {
		return nil, fmt.Errorf("failed to create kuzu database: %w", err)
	}
	conn, err := kuzudb.OpenConnection(db)
	if err != nil {
		db.Close()
		return nil, fmt.Errorf("failed to create kuzu connection: %w", err)
	}

	k := &Kuzu{
		DB:            db,
		Conn:          conn,
		topK:          topK,
		embeddingFunc: embeddingFunc,
	}

	// --- Schema and Index Initialization ---
	// Use conn.Query for one-off DDL statements without parameters.
	ddlQueries := []string{
		fmt.Sprintf("CREATE NODE TABLE IF NOT EXISTS Entity (name STRING, content STRING, embedding FLOAT[%d], PRIMARY KEY (name))", embeddingDim),
		fmt.Sprintf("CREATE REL TABLE IF NOT EXISTS Relationship (FROM Entity TO Entity, content STRING, embedding FLOAT[%d])", embeddingDim),
		"CREATE INDEX IF NOT EXISTS entity_embedding_idx ON Entity(embedding) USING HNSW",
		"CREATE INDEX IF NOT EXISTS rel_embedding_idx ON Relationship(embedding) USING HNSW",
	}

	for _, ddlQuery := range ddlQueries {
		q, err := conn.Query(ddlQuery)
		if err != nil {
			// KuzuDB doesn't have "CREATE INDEX IF NOT EXISTS" so we check for the error string.
			if strings.Contains(err.Error(), "already exists") {
				continue
			}
			k.Close()
			return nil, fmt.Errorf("failed to execute DDL query '%s': %w", ddlQuery, err)
		}
		q.Close()
	}

	// --- Prepare Statements ---
	// Prepare all recurring queries to be used later.
	err = k.prepareStatements()
	if err != nil {
		k.Close()
		return nil, fmt.Errorf("failed to prepare statements: %w", err)
	}

	return k, nil
}

// prepareStatements compiles the Cypher queries and stores them in the Kuzu struct.
func (k *Kuzu) prepareStatements() (err error) {
	queryEntity := `
        MATCH (e:Entity)
        RETURN e.name
        ORDER BY vector_similarity(e.embedding, $query_embedding) DESC
        LIMIT $topK`
	k.stmtQueryEntity, err = k.Conn.Prepare(queryEntity)
	if err != nil {
		return fmt.Errorf("failed to prepare query entity statement: %w", err)
	}

	queryRelationship := `
        MATCH (s:Entity)-[r:Relationship]->(t:Entity)
        RETURN s.name, t.name
        ORDER BY vector_similarity(r.embedding, $query_embedding) DESC
        LIMIT $topK`
	k.stmtQueryRelationship, err = k.Conn.Prepare(queryRelationship)
	if err != nil {
		return fmt.Errorf("failed to prepare query relationship statement: %w", err)
	}

	upsertEntity := "MERGE (e:Entity {name: $name}) SET e.content = $content, e.embedding = $embedding"
	k.stmtUpsertEntity, err = k.Conn.Prepare(upsertEntity)
	if err != nil {
		return fmt.Errorf("failed to prepare upsert entity statement: %w", err)
	}

	upsertRelationship := `
        MATCH (s:Entity {name: $source}), (t:Entity {name: $target})
        MERGE (s)-[r:Relationship]->(t)
        SET r.content = $content, r.embedding = $embedding`
	k.stmtUpsertRelationship, err = k.Conn.Prepare(upsertRelationship)
	if err != nil {
		return fmt.Errorf("failed to prepare upsert relationship statement: %w", err)
	}

	return nil
}

// Close gracefully destroys prepared statements and closes the database connection.
func (k *Kuzu) Close() {

	if k.Conn != nil {
		k.Conn.Close()
	}
	if k.DB != nil {
		k.DB.Close()
	}
}

// VectorQueryEntity performs a semantic search for entities using a prepared statement.
func (k *Kuzu) VectorQueryEntity(keywords string) ([]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	queryEmbedding, err := k.embeddingFunc(ctx, keywords)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for keywords: %w", err)
	}

	params := map[string]any{
		"query_embedding": queryEmbedding,
		"topK":            int64(k.topK),
	}

	queryResult, err := k.Conn.Execute(k.stmtQueryEntity, params)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query entities: %w", err)
	}
	defer queryResult.Close()

	var results []string
	for queryResult.HasNext() {
		row, err := queryResult.Next()
		slice, _ := row.GetAsSlice()
		if err != nil {
			return nil, fmt.Errorf("failed to retrieve query result row: %w", err)
		}
		if name, ok := slice[0].(string); ok {
			results = append(results, name)
		}
	}
	return results, nil
}

// VectorQueryRelationship performs a semantic search for relationships using a prepared statement.
func (k *Kuzu) VectorQueryRelationship(keywords string) ([][2]string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	queryEmbedding, err := k.embeddingFunc(ctx, keywords)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding for keywords: %w", err)
	}

	params := map[string]any{
		"query_embedding": queryEmbedding,
		"topK":            int64(k.topK),
	}

	queryResult, err := k.Conn.Execute(k.stmtQueryRelationship, params)
	if err != nil {
		return nil, fmt.Errorf("failed to execute query relationships: %w", err)
	}
	defer queryResult.Close()

	var results [][2]string
	for queryResult.HasNext() {
		row, err := queryResult.Next()
		slice, _ := row.GetAsSlice()
		if err != nil {
			return nil, fmt.Errorf("failed to retrieve query result row: %w", err)
		}
		source, s_ok := slice[0].(string)
		target, t_ok := slice[1].(string)
		if s_ok && t_ok {
			results = append(results, [2]string{source, target})
		}
	}
	return results, nil
}

// VectorUpsertEntity creates or updates an entity using a prepared statement.
func (k *Kuzu) VectorUpsertEntity(name, content string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	embedding, err := k.embeddingFunc(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding for entity '%s': %w", name, err)
	}

	params := map[string]any{
		"name":      name,
		"content":   content,
		"embedding": embedding,
	}

	queryResult, err := k.Conn.Execute(k.stmtUpsertEntity, params)
	if err != nil {
		return fmt.Errorf("failed to execute upsert entity: %w", err)
	}
	queryResult.Close()
	return nil
}

// VectorUpsertRelationship creates or updates a relationship using a prepared statement.
func (k *Kuzu) VectorUpsertRelationship(source, target, content string) error {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	embedding, err := k.embeddingFunc(ctx, content)
	if err != nil {
		return fmt.Errorf("failed to generate embedding for relationship '%s -> %s': %w", source, target, err)
	}

	params := map[string]any{
		"source":    source,
		"target":    target,
		"content":   content,
		"embedding": embedding,
	}

	queryResult, err := k.Conn.Execute(k.stmtUpsertRelationship, params)
	if err != nil {
		return fmt.Errorf("failed to execute upsert relationship: %w", err)
	}
	queryResult.Close()
	return nil
}
