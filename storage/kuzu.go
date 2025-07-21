package storage

import (
	"fmt"
	"strings"
	"time"

	golightrag "github.com/MegaGrindStone/go-light-rag"
	kuzu "github.com/kuzudb/go-kuzu"
)

// Kuzu provides a Kuzu graph database implementation of storage interfaces.
// It handles database connections and operations for storing and retrieving graph entities
// and relationships.
type Kuzu struct {
	DB   *kuzu.Database
	Conn *kuzu.Connection
}

// NewKuzu creates a new Kuzu client connection with the provided database path.
// It returns an initialized Kuzu struct and any error encountered during setup.
// The returned Kuzu instance must be closed with Close() when no longer needed.
func NewKuzu(dbPath string, systemConfig kuzu.SystemConfig) (Kuzu, error) {
	db, err := kuzu.OpenDatabase(dbPath, systemConfig)
	if err != nil {
		return Kuzu{}, fmt.Errorf("failed to create kuzu database: %w", err)
	}

	conn, err := kuzu.OpenConnection(db)
	if err != nil {
		db.Close() // Clean up the database if connection fails
		return Kuzu{}, fmt.Errorf("failed to create kuzu connection: %w", err)
	}

	k := Kuzu{DB: db, Conn: conn}

	if err := k.SetupSchema(); err != nil {
		// Clean up both on schema failure
		conn.Close()
		db.Close()
		return Kuzu{}, fmt.Errorf("failed to set up schema: %w", err)
	}

	return k, nil
}

// SetupSchema defines and creates the necessary node and relationship tables in Kuzu.
// This is idempotent; it will not fail if the tables already exist.
func (k Kuzu) SetupSchema() error {
	// Define the node table. entity_id is the primary key.
	nodeTableQuery := `
    CREATE NODE TABLE IF NOT EXISTS base (
        entity_id STRING,
        entity_type STRING,
        description STRING,
        source_ids STRING,
        created_at STRING,
        PRIMARY KEY (entity_id)
    )`
	// Define the relationship table.
	relTableQuery := `
    CREATE REL TABLE IF NOT EXISTS DIRECTED (
        FROM base TO base,
        weight DOUBLE,
        description STRING,
        keywords STRING,
        source_ids STRING,
        created_at STRING
    )`

	noteStmt, err := k.Conn.Query(nodeTableQuery)
	if err != nil {
		return fmt.Errorf("failed to execute create base node table: %w", err)
	}
	defer noteStmt.Close()

	relStmt, err := k.Conn.Query(relTableQuery)
	if err != nil {
		return fmt.Errorf("failed to prepare create rel table statement: %w", err)
	}
	defer relStmt.Close()

	return err
}

func graphEntityFromMap(props map[string]any) golightrag.GraphEntity {
	name, _ := props["entity_id"].(string)
	typ, _ := props["entity_type"].(string)
	desc, _ := props["description"].(string)
	sourceIDs, _ := props["source_ids"].(string)
	createdAtStr, _ := props["created_at"].(string)
	createdAt, err := time.Parse(time.RFC3339, createdAtStr)
	if err != nil {
		createdAt = time.Now()
	}

	return golightrag.GraphEntity{
		Name:         name,
		Type:         typ,
		Descriptions: desc,
		SourceIDs:    sourceIDs,
		CreatedAt:    createdAt,
	}
}

func graphRelationshipFromMap(source, target string, props map[string]any) golightrag.GraphRelationship {
	weight, _ := props["weight"].(float64)
	description, _ := props["description"].(string)
	keywords, _ := props["keywords"].(string)
	arrKeywords := strings.Split(keywords, golightrag.GraphFieldSeparator)
	sourceIDs, _ := props["source_ids"].(string)
	createdAtStr, _ := props["created_at"].(string)
	createdAt, err := time.Parse(time.RFC3339, createdAtStr)
	if err != nil {
		createdAt = time.Now()
	}

	return golightrag.GraphRelationship{
		SourceEntity: source,
		TargetEntity: target,
		Weight:       weight,
		Descriptions: description,
		Keywords:     arrKeywords,
		SourceIDs:    sourceIDs,
		CreatedAt:    createdAt,
	}
}

// GraphEntity retrieves a graph entity by name from the Kuzu database.
func (k Kuzu) GraphEntity(name string) (golightrag.GraphEntity, error) {
	query := `MATCH (n:base {entity_id: $entityID}) RETURN n`
	params := map[string]any{"entityID": name}
	prepped, _ := k.Conn.Prepare(query)
	queryResult, err := k.Conn.Execute(prepped, params)
	if err != nil {
		return golightrag.GraphEntity{}, fmt.Errorf("failed to run GraphEntity query: %w", err)
	}
	defer queryResult.Close()

	if !queryResult.HasNext() {
		return golightrag.GraphEntity{}, golightrag.ErrEntityNotFound
	}
	row, err := queryResult.Next()
	if err != nil {
		return golightrag.GraphEntity{}, fmt.Errorf("failed to get GraphEntity result row: %w", err)
	}

	nodeVal, err := row.GetValue(0)
	if err != nil {
		return golightrag.GraphEntity{}, fmt.Errorf("failed to get GraphEntity node value: %w", err)
	}
	nodeProps, ok := nodeVal.(kuzu.Node)
	if !ok {
		return golightrag.GraphEntity{}, fmt.Errorf("invalid node type, got %T, Kuzu.Node", nodeVal)
	}
	return graphEntityFromMap(nodeProps.Properties), nil
}

// GraphRelationship retrieves a relationship between two entities from the Kuzu database.
func (k Kuzu) GraphRelationship(sourceEntity, targetEntity string) (golightrag.GraphRelationship, error) {
	query := `
MATCH (s:base {entity_id: $source_entity_id}) -[r]- (e:base {entity_id: $target_entity_id})
RETURN {
keywords: r.keywords,
weight: r.weight,
description: r.description,
created_at: r.created_at,
source_ids: r.source_ids
} as edge_properties
`
	params := map[string]any{
		"source_entity_id": sourceEntity,
		"target_entity_id": targetEntity,
	}
	prepped, _ := k.Conn.Prepare(query)
	queryResult, err := k.Conn.Execute(prepped, params)
	if err != nil {
		return golightrag.GraphRelationship{}, fmt.Errorf("failed to run GraphRelationship query: %w", err)
	}
	defer queryResult.Close()

	if !queryResult.HasNext() {
		return golightrag.GraphRelationship{}, golightrag.ErrRelationshipNotFound
	}
	row, err := queryResult.Next()
	if err != nil {
		return golightrag.GraphRelationship{}, fmt.Errorf("failed to get GraphRelationship result row: %w", err)
	}
	edgePropsVal, err := row.GetValue(0)
	if err != nil {
		return golightrag.GraphRelationship{}, fmt.Errorf("failed to get edge GraphRelationship properties value: %w", err)
	}
	props, ok := edgePropsVal.(map[string]any)
	if !ok {
		return golightrag.GraphRelationship{},
			fmt.Errorf("invalid edge_properties type, got %T, want map[string]any", edgePropsVal)
	}

	return graphRelationshipFromMap(sourceEntity, targetEntity, props), nil
}

// GraphUpsertEntity creates or updates an entity in the Kuzu graph database.
func (k Kuzu) GraphUpsertEntity(entity golightrag.GraphEntity) error {
	query := `
MERGE (n:base {entity_id: $entity_id})
ON CREATE SET n.entity_type = $entity_type, n.source_ids = $source_ids, n.description = $description, n.created_at = $created_at
ON MATCH SET n.entity_type = $entity_type, n.source_ids = $source_ids, n.description = $description, n.created_at = $created_at
`
	params := map[string]any{
		"entity_id":   entity.Name,
		"entity_type": entity.Type,
		"description": entity.Descriptions,
		"source_ids":  entity.SourceIDs,
		"created_at":  entity.CreatedAt.Format(time.RFC3339),
	}
	prepped, err := k.Conn.Prepare(query)
	if err != nil {
		return fmt.Errorf("failed to prepare GraphUpsertEntity: %w", err)
	}
	_, err = k.Conn.Execute(prepped, params)
	return err
}

// GraphUpsertRelationship creates or updates a relationship between two entities.
func (k Kuzu) GraphUpsertRelationship(relationship golightrag.GraphRelationship) error {
	query := `
MATCH (source:base {entity_id: $source_entity_id})
WITH source
MATCH (target:base {entity_id: $target_entity_id})
MERGE (source)<-[r:DIRECTED]-(target)
ON CREATE SET  r.weight = $weight, r.description = $description, r.keywords = $keywords, r.source_ids = $source_ids, r.created_at = $created_at
ON MATCH SET r.weight = $weight, r.description = $description, r.keywords = $keywords, r.source_ids = $source_ids, r.created_at = $created_at
MERGE (target)-[r2:DIRECTED]->(source)
ON CREATE SET r2.weight = $weight, r2.description = $description, r2.keywords = $keywords, r2.source_ids = $source_ids, r2.created_at = $created_at
ON MATCH SET  r2.weight = $weight, r2.description = $description, r2.keywords = $keywords, r2.source_ids = $source_ids, r2.created_at = $created_at
`
	params := map[string]any{
		"source_entity_id": relationship.SourceEntity,
		"target_entity_id": relationship.TargetEntity,
		"weight":           relationship.Weight,
		"description":      relationship.Descriptions,
		"keywords":         strings.Join(relationship.Keywords, golightrag.GraphFieldSeparator),
		"source_ids":       relationship.SourceIDs,
		"created_at":       relationship.CreatedAt.Format(time.RFC3339),
	}
	prepped, _ := k.Conn.Prepare(query)
	_, err := k.Conn.Execute(prepped, params)
	return err
}

// GraphEntities retrieves multiple graph entities by their names from the Kuzu database.
func (k Kuzu) GraphEntities(names []string) (map[string]golightrag.GraphEntity, error) {
	if len(names) == 0 {
		return map[string]golightrag.GraphEntity{}, nil
	}

	query := `
	MATCH (n:base) 
	WHERE n.entity_id IN $entityIDs 
	RETURN n, n.entity_id as entity_id
	`
	params := map[string]any{"entityIDs": names}
	prepped, _ := k.Conn.Prepare(query)

	queryResult, err := k.Conn.Execute(prepped, params)
	if err != nil {
		return nil, fmt.Errorf("failed to run GraphUpsertRelationship query: %w", err)
	}
	defer queryResult.Close()

	entities := make(map[string]golightrag.GraphEntity)
	for queryResult.HasNext() {
		row, err := queryResult.Next()
		if err != nil {
			return nil, fmt.Errorf("failed to get GraphUpsertRelationship result row: %w", err)
		}
		nodeVal, err := row.GetValue(0)
		if err != nil {
			continue
		}
		nodeProps, ok := nodeVal.(map[string]any)
		if !ok {
			continue
		}
		entity := graphEntityFromMap(nodeProps)
		entities[entity.Name] = entity
	}

	return entities, nil
}

// GraphRelationships retrieves multiple relationships between entity pairs.
func (k Kuzu) GraphRelationships(pairs [][2]string) (map[string]golightrag.GraphRelationship, error) {
	if len(pairs) == 0 {
		return map[string]golightrag.GraphRelationship{}, nil
	}

	query := `
UNWIND $pairs AS pair
MATCH (s:base {entity_id: pair[1]})-[r]-(e:base {entity_id: pair[2]})
RETURN pair[1] as source, pair[2] as target, {
keywords: r.keywords,
weight: r.weight,
description: r.description,
created_at: r.created_at,
source_ids: r.source_ids
} as edge_properties
`
	pairsParam := make([][]string, len(pairs))
	for i, p := range pairs {
		pairsParam[i] = []string{p[0], p[1]}
	}
	params := map[string]any{"pairs": pairsParam}
	prepped, _ := k.Conn.Prepare(query)

	queryResult, err := k.Conn.Execute(prepped, params)
	if err != nil {
		return nil, fmt.Errorf("failed to run query: %w", err)
	}
	defer queryResult.Close()

	relationships := make(map[string]golightrag.GraphRelationship)
	for queryResult.HasNext() {
		row, err := queryResult.Next()
		if err != nil {
			return nil, fmt.Errorf("failed to get result row: %w", err)
		}
		sourceVal, _ := row.GetValue(0)
		targetVal, _ := row.GetValue(1)
		propsVal, _ := row.GetValue(2)

		sourceStr, sourceOK := sourceVal.(string)
		targetStr, targetOK := targetVal.(string)
		props, propsOK := propsVal.(map[string]any)

		if !sourceOK || !targetOK || !propsOK {
			continue
		}

		key := fmt.Sprintf("%s-%s", sourceStr, targetStr)
		relationships[key] = graphRelationshipFromMap(sourceStr, targetStr, props)
	}
	return relationships, nil
}

// GraphCountEntitiesRelationships counts the number of relationships for multiple entities.
func (k Kuzu) GraphCountEntitiesRelationships(names []string) (map[string]int, error) {
	if len(names) == 0 {
		return map[string]int{}, nil
	}

	query := `
MATCH (n:base)
WHERE n.entity_id IN $entity_ids
OPTIONAL MATCH (n)-[r]-()
RETURN n.entity_id AS entity_id, COUNT(r) AS degree
`
	params := map[string]any{"entity_ids": names}
	prepped, _ := k.Conn.Prepare(query)

	queryResult, err := k.Conn.Execute(prepped, params)
	if err != nil {
		return nil, fmt.Errorf("failed to run GraphCountEntitiesRelationships query: %w", err)
	}
	defer queryResult.Close()

	counts := make(map[string]int)
	for queryResult.HasNext() {
		row, err := queryResult.Next()
		if err != nil {
			return nil, fmt.Errorf("failed to get result GraphCountEntitiesRelationships row: %w", err)
		}
		entityIDVal, _ := row.GetValue(0)
		degreeVal, _ := row.GetValue(1)

		entityID, idOK := entityIDVal.(string)
		degree, degreeOK := degreeVal.(int64)

		if idOK && degreeOK {
			counts[entityID] = int(degree)
		}
	}
	return counts, nil
}

// GraphRelatedEntities retrieves all entities related to multiple input entities.
func (k Kuzu) GraphRelatedEntities(names []string) (map[string][]golightrag.GraphEntity, error) {
	if len(names) == 0 {
		return map[string][]golightrag.GraphEntity{}, nil
	}
	query := `
MATCH (n:base)
WHERE n.entity_id IN $entity_ids
OPTIONAL MATCH (n)-[r]-(connected:base)
WHERE connected.entity_id IS NOT NULL
RETURN n.entity_id as source_id, collect(connected) as connected_nodes
`
	params := map[string]any{"entity_ids": names}
	prepped, _ := k.Conn.Prepare(query)

	queryResult, err := k.Conn.Execute(prepped, params)
	if err != nil {
		return nil, fmt.Errorf("failed to run GraphRelatedEntities query: %w", err)
	}
	defer queryResult.Close()

	relatedEntities := make(map[string][]golightrag.GraphEntity)
	for queryResult.HasNext() {
		row, err := queryResult.Next()
		if err != nil {
			return nil, fmt.Errorf("failed to get GraphRelatedEntities result row: %w", err)
		}
		sourceIDVal, _ := row.GetValue(0)
		connectedNodesVal, _ := row.GetValue(1)

		sourceID, sourceOK := sourceIDVal.(string)
		connectedNodes, nodesOK := connectedNodesVal.([]any)

		if !sourceOK || !nodesOK {
			continue
		}

		entities := make([]golightrag.GraphEntity, 0, len(connectedNodes))
		for _, node := range connectedNodes {
			if nodeProps, ok := node.(map[string]any); ok {
				entities = append(entities, graphEntityFromMap(nodeProps))
			}
		}
		relatedEntities[sourceID] = entities
	}
	return relatedEntities, nil
}

// Close terminates the connection to the Kuzu database.
func (k *Kuzu) Close() {
	if k.Conn != nil {
		k.Conn.Close()
	}
	if k.DB != nil {
		k.DB.Close()
	}
}
