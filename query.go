package golightrag

import (
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"time"
)

// QueryHandler defines the interface for handling RAG query operations.
// It provides access to keyword extraction prompt data and the LLM.
type QueryHandler interface {
	// KeywordExtractionPromptData returns the data needed to generate prompts for extracting
	// keywords from user queries and conversation history.
	// The implementation doesn't need to fill the Query and History fields, as they will be filled
	// in the Query function.
	KeywordExtractionPromptData() KeywordExtractionPromptData
	LLM() LLM
}

// QueryConversation represents a message in a conversation with its role.
type QueryConversation struct {
	Message string
	Role    string
}

// QueryResult contains the retrieved context from both global and local searches.
// It includes entities, relationships, and sources organized by context type.
type QueryResult struct {
	GlobalEntities      []EntityContext
	GlobalRelationships []RelationshipContext
	GlobalSources       []SourceContext
	LocalEntities       []EntityContext
	LocalRelationships  []RelationshipContext
	LocalSources        []SourceContext
}

// EntityContext represents an entity retrieved from the knowledge graph with its context.
type EntityContext struct {
	Name        string
	Type        string
	Description string
	RefCount    int
	CreatedAt   time.Time
}

// RelationshipContext represents a relationship between entities retrieved from the knowledge graph.
type RelationshipContext struct {
	Source      string
	Target      string
	Keywords    string
	Description string
	Weight      float64
	RefCount    int
	CreatedAt   time.Time
}

// SourceContext represents a source document chunk with reference count.
type SourceContext struct {
	Content  string
	RefCount int
}

type keywordExtractionOutput struct {
	HighLevelKeywords []string `json:"high_level_keywords"`
	LowLevelKeywords  []string `json:"low_level_keywords"`
}

type refContext struct {
	context  string
	refCount int
}

const (
	// RoleUser represents the user role in a conversation.
	RoleUser = "user"
	// RoleAssistant represents the assistant role in a conversation.
	RoleAssistant = "assistant"
)

// Query performs a RAG search using the provided conversations.
// It extracts keywords from the user's query, searches for relevant entities and relationships
// in both local and global contexts, and returns the combined results.
func Query(
	conversations []QueryConversation,
	handler QueryHandler,
	storage Storage,
	logger *slog.Logger,
) (QueryResult, error) {
	logger = logger.With(
		slog.String("package", "golightrag"),
		slog.String("function", "Query"),
	)

	query, histories, err := extractQueryAndHistories(conversations)
	if err != nil {
		return QueryResult{}, fmt.Errorf("failed to extract query and histories: %w", err)
	}

	logger.Info("Extracted query", "query", query, "histories", histories)

	keywordData := handler.KeywordExtractionPromptData()
	keywordData.Query = query
	historiesStr := make([]string, len(histories))
	for i, history := range histories {
		historiesStr[i] = history.String()
	}
	keywordData.History = strings.Join(historiesStr, "\n")

	keywordPrompt, err := promptTemplate("extract-keywords", keywordExtractionPrompt, keywordData)
	if err != nil {
		return QueryResult{}, fmt.Errorf("failed to generate keyword extraction prompt: %w", err)
	}

	logger.Debug("Use LLM to extract keywords from query", "keywordPrompt", keywordPrompt)

	llm := handler.LLM()

	keywordRes, err := llm.Chat([]string{keywordPrompt})
	if err != nil {
		return QueryResult{}, fmt.Errorf("failed to call LLM: %w", err)
	}

	logger.Debug("Extracted keywords from LLM", "keywords", keywordRes)

	var output keywordExtractionOutput
	err = json.Unmarshal([]byte(keywordRes), &output)
	if err != nil {
		return QueryResult{}, fmt.Errorf("failed to unmarshal keyword extraction output: %w", err)
	}

	logger.Info("Query keywords",
		"highLevelKeywords", output.HighLevelKeywords,
		"lowLevelKeywords", output.LowLevelKeywords,
	)

	llKeywords := strings.Join(output.LowLevelKeywords, ", ")
	hlKeywords := strings.Join(output.HighLevelKeywords, ", ")

	localEntities, localRelationships, localSources, err := localContext(llKeywords, storage, logger)
	if err != nil {
		return QueryResult{}, fmt.Errorf("failed to get local context: %w", err)
	}

	globalEntities, globalRelationships, globalSources, err := globalContext(hlKeywords, storage, logger)
	if err != nil {
		return QueryResult{}, fmt.Errorf("failed to get global context: %w", err)
	}

	return QueryResult{
		LocalEntities:       localEntities,
		LocalRelationships:  localRelationships,
		LocalSources:        localSources,
		GlobalEntities:      globalEntities,
		GlobalRelationships: globalRelationships,
		GlobalSources:       globalSources,
	}, nil
}

func extractQueryAndHistories(conversations []QueryConversation) (string, []QueryConversation, error) {
	for i := len(conversations) - 1; i >= 0; i-- {
		if conversations[i].Role == RoleUser {
			return conversations[i].Message, conversations[:i], nil
		}
	}

	return "", nil, errors.New("no user message found")
}

func localContext(
	keywords string,
	storage Storage,
	logger *slog.Logger,
) ([]EntityContext, []RelationshipContext, []SourceContext, error) {
	entitiesNames, err := storage.VectorQueryEntity(keywords)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to query entities: %w", err)
	}

	if len(entitiesNames) == 0 {
		return []EntityContext{}, []RelationshipContext{}, []SourceContext{}, nil
	}

	logger.Debug("Entities names from vector storage", "entitiesNames", entitiesNames)

	var entities []GraphEntity
	var entitiesContexts []EntityContext

	for _, name := range entitiesNames {
		entity, err := storage.GraphEntity(name)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to get entity with name %s: %w", name, err)
		}

		entities = append(entities, entity)

		refCount, err := storage.GraphCountEntityRelationships(name)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to count entity relationships with name %s: %w", name, err)
		}

		entitiesContexts = append(entitiesContexts, EntityContext{
			Name:        entity.Name,
			Type:        entity.Type,
			Description: entity.Descriptions,
			RefCount:    refCount,
			CreatedAt:   entity.CreatedAt,
		})
	}

	logger.Debug("Entities from graph storage", "entities", entities)

	rankedRelationships, err := entitiesRankedRelationship(entities, storage)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get ranked relationships: %w", err)
	}

	rankedSources, err := entitiesRankedSources(entities, storage)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get ranked sources: %w", err)
	}

	return entitiesContexts, rankedRelationships, rankedSources, nil
}

func globalContext(
	keywords string,
	storage Storage,
	logger *slog.Logger,
) ([]EntityContext, []RelationshipContext, []SourceContext, error) {
	relationshipNames, err := storage.VectorQueryRelationship(keywords)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to query relationships: %w", err)
	}

	if len(relationshipNames) == 0 {
		return []EntityContext{}, []RelationshipContext{}, []SourceContext{}, nil
	}

	logger.Debug("Relationship names from vector storage", "relationshipNames", relationshipNames)

	var relationships []GraphRelationship
	var relationshipsContexts []RelationshipContext

	for _, rel := range relationshipNames {
		sourceEntity, targetEntity := rel[0], rel[1]

		relationship, err := storage.GraphRelationship(sourceEntity, targetEntity)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to get relationship for source entity %s and target entity %s: %w",
				sourceEntity, targetEntity, err)
		}

		relationships = append(relationships, relationship)

		sourceDegree, err := storage.GraphCountEntityRelationships(sourceEntity)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to count entity relationships with name %s: %w", sourceEntity, err)
		}
		targetDegree, err := storage.GraphCountEntityRelationships(targetEntity)
		if err != nil {
			return nil, nil, nil, fmt.Errorf("failed to count entity relationships with name %s: %w", targetEntity, err)
		}

		refCount := sourceDegree + targetDegree
		relationshipsContexts = append(relationshipsContexts, RelationshipContext{
			Source:      sourceEntity,
			Target:      targetEntity,
			Keywords:    relationship.Keywords,
			Description: relationship.Descriptions,
			Weight:      relationship.Weight,
			RefCount:    refCount,
			CreatedAt:   relationship.CreatedAt,
		})
	}

	logger.Debug("Relationships from graph storage", "relationships", relationships)

	rankedEntities, err := relationshipsRankedEntities(relationships, storage)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get ranked entities: %w", err)
	}

	rankedSources, err := relationshipsRankedSources(relationships, storage)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get ranked sources: %w", err)
	}

	return rankedEntities, relationshipsContexts, rankedSources, nil
}

func entitiesRankedRelationship(entities []GraphEntity, storage Storage) ([]RelationshipContext, error) {
	relationshipCountMap := make(map[string]int)
	relationshipMap := make(map[string]GraphRelationship)
	for _, entity := range entities {
		relEntities, err := storage.GraphRelatedEntities(entity.Name)
		if err != nil {
			return nil, fmt.Errorf("failed to get related entities for entity %s: %w", entity.Name, err)
		}
		for _, relEntity := range relEntities {
			rel, err := storage.GraphRelationship(entity.Name, relEntity.Name)
			if err != nil {
				return nil, fmt.Errorf("failed to get relationship for entity %s and %s: %w", entity.Name, relEntity.Name, err)
			}

			srcDegree, err := storage.GraphCountEntityRelationships(entity.Name)
			if err != nil {
				return nil, fmt.Errorf("failed to count entity relationships with name %s: %w", entity.Name, err)
			}
			tgtDegree, err := storage.GraphCountEntityRelationships(relEntity.Name)
			if err != nil {
				return nil, fmt.Errorf("failed to count entity relationships with name %s: %w", relEntity.Name, err)
			}

			key := fmt.Sprintf("%s-%s", rel.SourceEntity, rel.TargetEntity)
			relationshipMap[key] = rel
			relationshipCountMap[key] = srcDegree + tgtDegree
		}
	}

	result := make([]RelationshipContext, 0, len(relationshipCountMap))
	for key, count := range relationshipCountMap {
		rel := relationshipMap[key]
		result = append(result, RelationshipContext{
			Source:      rel.SourceEntity,
			Target:      rel.TargetEntity,
			Keywords:    rel.Keywords,
			Description: rel.Descriptions,
			Weight:      rel.Weight,
			RefCount:    count,
			CreatedAt:   rel.CreatedAt,
		})
	}

	return result, nil
}

func entitiesRankedSources(entities []GraphEntity, storage Storage) ([]SourceContext, error) {
	sourceIDCountMap := make(map[string]int)
	for _, entity := range entities {
		arrSourceID := strings.Split(entity.SourceIDs, graphFieldSeparator)
		for _, sourceID := range arrSourceID {
			if sourceID == "" {
				continue
			}
			_, ok := sourceIDCountMap[sourceID]
			if ok {
				continue
			}
			sourceIDCountMap[sourceID] = 0
		}
	}

	for _, entity := range entities {
		relEntities, err := storage.GraphRelatedEntities(entity.Name)
		if err != nil {
			return nil, fmt.Errorf("failed to get related entities for entity %s: %w", entity.Name, err)
		}
		for _, relEntity := range relEntities {
			arrSourceID := strings.Split(relEntity.SourceIDs, graphFieldSeparator)
			for _, sourceID := range arrSourceID {
				if sourceID == "" {
					continue
				}
				_, ok := sourceIDCountMap[sourceID]
				if ok {
					sourceIDCountMap[sourceID]++
				}
			}
		}
	}

	result := make([]SourceContext, 0, len(sourceIDCountMap))
	for id, count := range sourceIDCountMap {
		source, err := storage.KVSource(id)
		if err != nil {
			return nil, fmt.Errorf("failed to get source with id %s: %w", id, err)
		}
		result = append(result, SourceContext{
			Content:  source.Content,
			RefCount: count,
		})
	}

	return result, nil
}

func relationshipsRankedEntities(relationships []GraphRelationship, storage Storage) ([]EntityContext, error) {
	entityNames := make([]string, 0, len(relationships))
	for _, rel := range relationships {
		entityNames = appendIfUnique(entityNames, rel.SourceEntity)
		entityNames = appendIfUnique(entityNames, rel.TargetEntity)
	}

	entities := make([]EntityContext, 0, len(entityNames))
	for _, entityName := range entityNames {
		if entityName == "" {
			continue
		}

		entity, err := storage.GraphEntity(entityName)
		if err != nil {
			return nil, fmt.Errorf("failed to get entity with name %s: %w", entityName, err)
		}
		degree, err := storage.GraphCountEntityRelationships(entityName)
		if err != nil {
			return nil, fmt.Errorf("failed to count entity relationships with name %s: %w", entityName, err)
		}
		entities = append(entities, EntityContext{
			Name:        entity.Name,
			Type:        entity.Type,
			Description: entity.Descriptions,
			RefCount:    degree,
		})
	}

	return entities, nil
}

func relationshipsRankedSources(relationships []GraphRelationship, storage Storage) ([]SourceContext, error) {
	sourcesMap := make(map[string]SourceContext)
	for _, rel := range relationships {
		arrSourceIDs := strings.Split(rel.SourceIDs, graphFieldSeparator)
		for _, sourceID := range arrSourceIDs {
			if sourceID == "" {
				continue
			}
			source, err := storage.KVSource(sourceID)
			if err != nil {
				return nil, fmt.Errorf("failed to get source with id %s: %w", sourceID, err)
			}
			_, ok := sourcesMap[sourceID]
			if !ok {
				sourcesMap[sourceID] = SourceContext{
					Content: source.Content,
				}
			}
			src := sourcesMap[sourceID]
			src.RefCount++
			sourcesMap[sourceID] = src
		}
	}

	sources := make([]SourceContext, len(sourcesMap))
	i := 0
	for _, source := range sourcesMap {
		sources[i] = source
		i++
	}

	return sources, nil
}

func combineContexts(headers []string, ctx1, ctx2 []refContext) string {
	resMap := make(map[string]refContext)
	for _, ctx := range ctx1 {
		resMap[ctx.context] = ctx
	}
	for _, ctx := range ctx2 {
		resMap[ctx.context] = ctx
	}

	arrRes := make([]refContext, 0, len(resMap))
	for _, ctx := range resMap {
		arrRes = append(arrRes, ctx)
	}

	slices.SortFunc(arrRes, func(a, b refContext) int {
		return cmp.Compare(b.refCount, a.refCount)
	})

	res := strings.Join(headers, ",") + "\n"
	for i, ctx := range arrRes {
		idStr := strconv.Itoa(i)
		res += fmt.Sprintf("%q,%s\n", idStr, ctx.context)
	}

	return res
}

// String returns a string representation of the QueryConversation showing its role and content.
func (q QueryConversation) String() string {
	return fmt.Sprintf("role: %s, content: %s", q.Role, q.Message)
}

// String returns a CSV-formatted string representation of the QueryResult with entities,
// relationships, and sources organized in sections.
func (q QueryResult) String() string {
	globalEntities := make([]refContext, len(q.GlobalEntities))
	for i, entity := range q.GlobalEntities {
		globalEntities[i] = refContext{
			context:  entity.String(),
			refCount: entity.RefCount,
		}
	}
	localEntities := make([]refContext, len(q.LocalEntities))
	for i, entity := range q.LocalEntities {
		localEntities[i] = refContext{
			context:  entity.String(),
			refCount: entity.RefCount,
		}
	}
	entities := combineContexts([]string{"id", "name", "type", "description", "ref_count", "created_at"},
		globalEntities, localEntities)

	globalRelationships := make([]refContext, len(q.GlobalRelationships))
	for i, relationship := range q.GlobalRelationships {
		globalRelationships[i] = refContext{
			context:  relationship.String(),
			refCount: relationship.RefCount,
		}
	}
	localRelationships := make([]refContext, len(q.LocalRelationships))
	for i, relationship := range q.LocalRelationships {
		localRelationships[i] = refContext{
			context:  relationship.String(),
			refCount: relationship.RefCount,
		}
	}
	relationships := combineContexts(
		[]string{"id", "source", "target", "keywords", "description", "weight", "ref_count", "created_at"},
		globalRelationships, localRelationships)

	globalSources := make([]refContext, len(q.GlobalSources))
	for i, source := range q.GlobalSources {
		globalSources[i] = refContext{
			context:  source.String(),
			refCount: source.RefCount,
		}
	}
	localSources := make([]refContext, len(q.LocalSources))
	for i, source := range q.LocalSources {
		localSources[i] = refContext{
			context:  source.String(),
			refCount: source.RefCount,
		}
	}
	sources := combineContexts([]string{"id", "content", "ref_count"}, globalSources, localSources)

	return fmt.Sprintf(`
-----Entities-----
`+threeBacktick("csv")+`
%s
`+threeBacktick("")+`
-----Relationships-----
`+threeBacktick("csv")+`
%s
`+threeBacktick("")+`
-----Sources-----
`+threeBacktick("csv")+`
%s
`+threeBacktick(""), entities, relationships, sources)
}

// String returns a CSV-formatted string representation of the EntityContext.
func (e EntityContext) String() string {
	refStr := strconv.Itoa(e.RefCount)
	return fmt.Sprintf("%q,%q,%q,%q,%q", e.Name, e.Type, e.Description, refStr, e.CreatedAt)
}

// String returns a CSV-formatted string representation of the RelationshipContext.
func (r RelationshipContext) String() string {
	weightStr := strconv.FormatFloat(r.Weight, 'f', 2, 64)
	refStr := strconv.Itoa(r.RefCount)
	return fmt.Sprintf("%q,%q,%q,%q,%q,%q,%q",
		r.Source, r.Target, r.Keywords, r.Description, weightStr, refStr, r.CreatedAt)
}

// String returns a CSV-formatted string representation of the SourceContext.
func (s SourceContext) String() string {
	refStr := strconv.Itoa(s.RefCount)
	return fmt.Sprintf("%q,%q", s.Content, refStr)
}
