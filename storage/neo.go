package storage

import (
	"context"
	"fmt"
	"time"

	golightrag "github.com/MegaGrindStone/go-light-rag"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j/db"
	"github.com/neo4j/neo4j-go-driver/v5/neo4j/dbtype"
)

// Neo4J provides a Neo4j graph database implementation of storage interfaces.
// It handles database connections and operations for storing and retrieving graph entities
// and relationships.
type Neo4J struct {
	client neo4j.DriverWithContext
}

// NewNeo4J creates a new Neo4j client connection with the provided connection parameters.
// It returns an initialized Neo4J struct and any error encountered during connection setup.
// The returned Neo4J instance must be closed with Close() when no longer needed to free up resources.
func NewNeo4J(target, user, password string) (Neo4J, error) {
	driver, err := neo4j.NewDriverWithContext(
		target,
		neo4j.BasicAuth(user, password, ""))
	if err != nil {
		return Neo4J{}, fmt.Errorf("failed to create neo4j driver: %w", err)
	}
	return Neo4J{client: driver}, nil
}

func graphEntityFromNode(node dbtype.Node) golightrag.GraphEntity {
	name, ok := node.Props["entity_id"].(string)
	if !ok {
		name = ""
	}
	typ, ok := node.Props["entity_type"].(string)
	if !ok {
		typ = ""
	}
	desc, ok := node.Props["description"].(string)
	if !ok {
		desc = ""
	}
	sourceIDs, ok := node.Props["source_ids"].(string)
	if !ok {
		sourceIDs = ""
	}
	createdAtStr, ok := node.Props["created_at"].(string)
	if !ok {
		createdAtStr = time.Now().Format(time.RFC3339)
	}
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

func graphRelationshipFromEdge(source, target string, props map[string]any) golightrag.GraphRelationship {
	weight, ok := props["weight"].(float64)
	if !ok {
		weight = 1.0
	}
	description, ok := props["description"].(string)
	if !ok {
		description = ""
	}
	keywords, ok := props["keywords"].(string)
	if !ok {
		keywords = ""
	}
	sourceIDs, ok := props["source_ids"].(string)
	if !ok {
		sourceIDs = ""
	}
	createdAtStr, ok := props["created_at"].(string)
	if !ok {
		createdAtStr = time.Now().Format(time.RFC3339)
	}
	createdAt, err := time.Parse(time.RFC3339, createdAtStr)
	if err != nil {
		createdAt = time.Now()
	}

	return golightrag.GraphRelationship{
		SourceEntity: source,
		TargetEntity: target,
		Weight:       weight,
		Descriptions: description,
		Keywords:     keywords,
		SourceIDs:    sourceIDs,
		CreatedAt:    createdAt,
	}
}

// GraphEntity retrieves a graph entity by name from the Neo4j database.
// It returns the found entity or an error if the entity doesn't exist or if the query fails.
func (n Neo4J) GraphEntity(name string) (golightrag.GraphEntity, error) {
	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := "MATCH (n:base {entity_id: $entityID}) RETURN n"
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"entityID": name,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			ety, err := queryRes.Single(ctx)
			if err != nil {
				return nil, golightrag.ErrEntityNotFound
			}
			return ety, nil
		})
	})
	if err != nil {
		return golightrag.GraphEntity{}, err
	}
	record, ok := res.(*db.Record)
	if !ok {
		return golightrag.GraphEntity{}, fmt.Errorf("invalid result type, got %T, want *db.Record", res)
	}
	nNode, ok := record.Get("n")
	if !ok {
		return golightrag.GraphEntity{}, fmt.Errorf("expected n key is not found")
	}
	node, ok := nNode.(dbtype.Node)
	if !ok {
		return golightrag.GraphEntity{}, fmt.Errorf("invalid n type, got %T, want dbtype.Node", n)
	}

	return graphEntityFromNode(node), nil
}

// GraphRelationship retrieves a relationship between two entities from the Neo4j database.
// It returns the found relationship or an error if the relationship doesn't exist or if the query fails.
func (n Neo4J) GraphRelationship(sourceEntity, targetEntity string) (golightrag.GraphRelationship, error) {
	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := `
MATCH (start:base {entity_id: $source_entity_id})-[r]-(end:base {entity_id: $target_entity_id})
RETURN properties(r) as edge_properties
      `
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"source_entity_id": sourceEntity,
				"target_entity_id": targetEntity,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			ety, err := queryRes.Single(ctx)
			if err != nil {
				return nil, golightrag.ErrRelationshipNotFound
			}
			return ety, nil
		})
	})
	if err != nil {
		return golightrag.GraphRelationship{}, err
	}
	record, ok := res.(*db.Record)
	if !ok {
		return golightrag.GraphRelationship{}, fmt.Errorf("invalid result type, got %T, want *db.Record", res)
	}
	edgeProps, ok := record.Get("edge_properties")
	if !ok {
		return golightrag.GraphRelationship{}, fmt.Errorf("expected edge_properties key is not found")
	}
	props, ok := edgeProps.(map[string]any)
	if !ok {
		return golightrag.GraphRelationship{},
			fmt.Errorf("invalid edge_properties type, got %T, want map[string]any", edgeProps)
	}

	return graphRelationshipFromEdge(sourceEntity, targetEntity, props), nil
}

// GraphUpsertEntity creates or updates an entity in the Neo4j graph database.
// It returns an error if the database operation fails.
func (n Neo4J) GraphUpsertEntity(entity golightrag.GraphEntity) error {
	_, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			return tx.Run(
				ctx,
				fmt.Sprintf(`
MERGE (n:base {entity_id: $properties.entity_id})
SET n += $properties
SET n:%s`, "`"+entity.Type+"`"),
				map[string]any{
					"properties": map[string]any{
						"entity_id":   entity.Name,
						"entity_type": entity.Type,
						"description": entity.Descriptions,
						"source_ids":  entity.SourceIDs,
						"created_at":  entity.CreatedAt.Format(time.RFC3339),
					},
				},
			)
		})
	})

	return err
}

// GraphUpsertRelationship creates or updates a relationship between two entities in the Neo4j graph database.
// It returns an error if the database operation fails.
func (n Neo4J) GraphUpsertRelationship(relationship golightrag.GraphRelationship) error {
	_, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteWrite(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			return tx.Run(
				ctx,
				`
MATCH (source:base {entity_id: $source_entity_id})
WITH source
MATCH (target:base {entity_id: $target_entity_id})
MERGE (source)-[r:DIRECTED]-(target)
SET r += $properties
`,
				map[string]any{
					"source_entity_id": relationship.SourceEntity,
					"target_entity_id": relationship.TargetEntity,
					"properties": map[string]any{
						"weight":      relationship.Weight,
						"description": relationship.Descriptions,
						"keywords":    relationship.Keywords,
						"source_ids":  relationship.SourceIDs,
						"created_at":  relationship.CreatedAt.Format(time.RFC3339),
					},
				},
			)
		})
	})

	return err
}

// GraphCountEntityRelationships counts the number of relationships connected to a specific entity.
// It returns the count of relationships and any error encountered during the operation.
func (n Neo4J) GraphCountEntityRelationships(name string) (int, error) {
	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := `
MATCH (n:base {entity_id: $entity_id})
OPTIONAL MATCH (n)-[r]-()
RETURN COUNT(r) AS degree`
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"entity_id": name,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			ety, err := queryRes.Single(ctx)
			if err != nil {
				return nil, fmt.Errorf("failed to get result: %w", err)
			}
			return ety.AsMap(), nil
		})
	})
	if err != nil {
		return 0, err
	}
	resMap, ok := res.(map[string]any)
	if !ok {
		return 0, fmt.Errorf("invalid result type: %T", res)
	}
	if len(resMap) == 0 {
		return 0, nil
	}

	degree, ok := resMap["degree"].(int64)
	if !ok {
		return 0, fmt.Errorf("invalid degree type: %T", resMap["degree"])
	}

	return int(degree), nil
}

// GraphRelatedEntities retrieves all entities that have a relationship with the specified entity.
// It returns a slice of related entities and any error encountered during the operation.
func (n Neo4J) GraphRelatedEntities(name string) ([]golightrag.GraphEntity, error) {
	res, err := n.session(func(ctx context.Context, sess neo4j.SessionWithContext) (any, error) {
		return sess.ExecuteRead(ctx, func(tx neo4j.ManagedTransaction) (any, error) {
			query := `
MATCH (n:base {entity_id: $entity_id})
OPTIONAL MATCH (n)-[r]-(connected:base)
WHERE connected.entity_id IS NOT NULL
RETURN n, r, connected`
			queryRes, err := tx.Run(ctx, query, map[string]any{
				"entity_id": name,
			})
			if err != nil {
				return nil, fmt.Errorf("failed to run query: %w", err)
			}

			nodes := make([]dbtype.Node, 0)
			for record, err := range queryRes.Records(ctx) {
				if err != nil {
					return nil, fmt.Errorf("failed to get result: %w", err)
				}

				r, ok := record.Get("connected")
				if !ok {
					return nil, fmt.Errorf("expected connected key is not found")
				}
				if r == nil {
					continue
				}
				resMap, ok := r.(dbtype.Node)
				if !ok {
					return nil, fmt.Errorf("invalid result type, got %T, want dbtype.Node", r)
				}

				nodes = append(nodes, resMap)
			}

			return nodes, nil
		})
	})
	if err != nil {
		return nil, err
	}
	nodes, ok := res.([]dbtype.Node)
	if !ok {
		return nil, fmt.Errorf("invalid result type: got %T, want []dbtype.Node", res)
	}

	entities := make([]golightrag.GraphEntity, 0)
	for _, node := range nodes {
		entities = append(entities, graphEntityFromNode(node))
	}

	return entities, nil
}

// Close terminates the connection to the Neo4j database.
// It returns any error encountered during the closing operation.
func (n Neo4J) Close(ctx context.Context) error {
	return n.client.Close(ctx)
}

func (n Neo4J) session(sessFunc func(context.Context, neo4j.SessionWithContext) (any, error)) (any, error) {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	sess := n.client.NewSession(ctx, neo4j.SessionConfig{})
	defer func() {
		closeCtx, closeCancel := context.WithTimeout(context.Background(), time.Second*30)
		defer closeCancel()
		_ = sess.Close(closeCtx)
	}()

	trxCtx, trxCancel := context.WithTimeout(context.Background(), time.Second*30)
	defer trxCancel()

	return sessFunc(trxCtx, sess)
}
