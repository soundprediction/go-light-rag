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
	"sync"
	"time"
)

// QueryHandler defines the interface for handling RAG query operations.
type QueryHandler interface {
	// KeywordExtractionPromptData returns the data needed to generate prompts for extracting
	// keywords from user queries and conversation history.
	// The implementation doesn't need to fill the Query and History fields, as they will be filled
	// in the Query function.
	KeywordExtractionPromptData() KeywordExtractionPromptData
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
	llm LLM,
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

	keywordRes, err := llm.Chat([]string{keywordPrompt})
	if err != nil {
		return QueryResult{}, fmt.Errorf("failed to call LLM: %w", err)
	}

	logger.Debug("Extracted keywords from LLM", "keywords", keywordRes)

	var output keywordExtractionOutput
	err = json.Unmarshal([]byte(strings.ReplaceAll(keywordRes, "\\", "")), &output)
	if err != nil {
		return QueryResult{}, fmt.Errorf("failed to unmarshal keyword extraction output: %w", err)
	}

	logger.Info("Query keywords",
		"highLevelKeywords", output.HighLevelKeywords,
		"lowLevelKeywords", output.LowLevelKeywords,
	)

	llKeywords := strings.Join(output.LowLevelKeywords, ", ")
	hlKeywords := strings.Join(output.HighLevelKeywords, ", ")

	// Run local and global context retrieval concurrently
	var localEntities []EntityContext
	var localRelationships []RelationshipContext
	var localSources []SourceContext
	var globalEntities []EntityContext
	var globalRelationships []RelationshipContext
	var globalSources []SourceContext
	var localErr, globalErr error

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer wg.Done()
		localEntities, localRelationships, localSources, localErr = localContext(llKeywords, storage, logger)
	}()

	go func() {
		defer wg.Done()
		globalEntities, globalRelationships, globalSources, globalErr = globalContext(hlKeywords, storage, logger)
	}()

	wg.Wait()

	if localErr != nil {
		return QueryResult{}, fmt.Errorf("failed to get local context: %w", localErr)
	}

	if globalErr != nil {
		return QueryResult{}, fmt.Errorf("failed to get global context: %w", globalErr)
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
	// First find relevant entities using vector similarity search
	entitiesNames, err := storage.VectorQueryEntity(keywords)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to query entities: %w", err)
	}

	if len(entitiesNames) == 0 {
		return []EntityContext{}, []RelationshipContext{}, []SourceContext{}, nil
	}

	logger.Debug("Entities names from vector storage", "entitiesNames", entitiesNames)

	// Get full entity details from graph storage
	entitiesMap, err := storage.GraphEntities(entitiesNames)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to batch get entities: %w", err)
	}
	// Get relationship counts to determine entity importance
	refCountMap, err := storage.GraphCountEntitiesRelationships(entitiesNames)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to batch count relationships: %w", err)
	}

	entitiesContexts := make([]EntityContext, 0, len(entitiesMap))
	entities := make([]GraphEntity, 0, len(entitiesMap))

	for name, entity := range entitiesMap {
		entities = append(entities, entity)
		refCount, ok := refCountMap[name]
		if !ok {
			refCount = 0
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

	// Get and rank relationships between the found entities
	rankedRelationships, err := entitiesRankedRelationships(entities, storage)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get ranked relationships: %w", err)
	}

	// Get and rank source documents referenced by the found entities
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
	// Start by querying relationships (unlike localContext which queries entities first)
	// This prioritizes connections between concepts rather than specific entities
	relationshipNames, err := storage.VectorQueryRelationship(keywords)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to query relationships: %w", err)
	}

	if len(relationshipNames) == 0 {
		return []EntityContext{}, []RelationshipContext{}, []SourceContext{}, nil
	}

	logger.Debug("Relationship names from vector storage", "relationshipNames", relationshipNames)

	// Get full details of the relationships from graph storage
	relationshipsMap, err := storage.GraphRelationships(relationshipNames)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to query relationships: %w", err)
	}

	// Collect all unique entity names that are part of these relationships
	entitiesNames := make([]string, 0, len(relationshipsMap))
	for _, rel := range relationshipsMap {
		entitiesNames = appendIfUnique(entitiesNames, rel.SourceEntity)
		entitiesNames = appendIfUnique(entitiesNames, rel.TargetEntity)
	}

	// Get relationship counts for relevance scoring
	refCountMap, err := storage.GraphCountEntitiesRelationships(entitiesNames)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to batch count relationships: %w", err)
	}

	relationships := make([]GraphRelationship, 0, len(relationshipsMap))
	relationshipsContexts := make([]RelationshipContext, 0, len(relationshipsMap))

	for _, rel := range relationshipsMap {
		relationships = append(relationships, rel)

		// Calculate importance score by summing relationship counts of both entities
		sourceDegree, ok := refCountMap[rel.SourceEntity]
		if !ok {
			sourceDegree = 0
		}
		targetDegree, ok := refCountMap[rel.TargetEntity]
		if !ok {
			targetDegree = 0
		}

		refCount := sourceDegree + targetDegree
		keywordsStr := strings.Join(rel.Keywords, GraphFieldSeparator)
		relationshipsContexts = append(relationshipsContexts, RelationshipContext{
			Source:      rel.SourceEntity,
			Target:      rel.TargetEntity,
			Keywords:    keywordsStr,
			Description: rel.Descriptions,
			Weight:      rel.Weight,
			RefCount:    refCount,
			CreatedAt:   rel.CreatedAt,
		})
	}

	logger.Debug("Relationships from graph storage", "relationships", relationships)

	// Get entities connected by these relationships (inverse of localContext flow)
	rankedEntities, err := relationshipsRankedEntities(relationships, storage)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get ranked entities: %w", err)
	}

	// Get source documents referenced by these relationships
	rankedSources, err := relationshipsRankedSources(relationships, storage)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to get ranked sources: %w", err)
	}

	return rankedEntities, relationshipsContexts, rankedSources, nil
}

func entitiesRankedRelationships(entities []GraphEntity, storage Storage) ([]RelationshipContext, error) {
	entityNames := make([]string, len(entities))
	for i, entity := range entities {
		entityNames[i] = entity.Name
	}

	// Get entities that are directly connected to our search results
	relationEntitiesMap, err := storage.GraphRelatedEntities(entityNames)
	if err != nil {
		return nil, fmt.Errorf("failed to get related entities: %w", err)
	}
	// Create pairs of entity names to retrieve their connecting relationships
	relationshipPairs := make([][2]string, 0)
	allEntities := make([]string, 0)
	allEntities = append(allEntities, entityNames...)
	for name, entities := range relationEntitiesMap {
		for _, entity := range entities {
			relationshipPairs = append(relationshipPairs, [2]string{name, entity.Name})
		}
		allEntities = appendIfUnique(allEntities, name)
	}

	// Fetch actual relationship data for all the entity pairs
	relationshipsMap, err := storage.GraphRelationships(relationshipPairs)
	if err != nil {
		return nil, fmt.Errorf("failed to query relationships: %w", err)
	}

	// Count relationships for relevance scoring
	refCountMap, err := storage.GraphCountEntitiesRelationships(allEntities)
	if err != nil {
		return nil, fmt.Errorf("failed to batch count relationships: %w", err)
	}

	result := make([]RelationshipContext, 0, len(relationshipsMap))
	for key, rel := range relationshipsMap {
		// Parse composite key back into source and target entities
		arrKey := strings.Split(key, "-")
		if len(arrKey) != 2 {
			continue
		}
		source := arrKey[0]
		target := arrKey[1]

		// Calculate importance by summing relationship counts of both entities
		sourceDegree, ok := refCountMap[source]
		if !ok {
			sourceDegree = 0
		}
		targetDegree, ok := refCountMap[target]
		if !ok {
			targetDegree = 0
		}

		refCount := sourceDegree + targetDegree
		keywordsStr := strings.Join(rel.Keywords, GraphFieldSeparator)
		result = append(result, RelationshipContext{
			Source:      source,
			Target:      target,
			Keywords:    keywordsStr,
			Description: rel.Descriptions,
			Weight:      rel.Weight,
			RefCount:    refCount,
			CreatedAt:   rel.CreatedAt,
		})
	}

	return result, nil
}

func entitiesRankedSources(entities []GraphEntity, storage Storage) ([]SourceContext, error) {
	entityNames := make([]string, len(entities))

	// Track sources and their reference counts across entities and relationships
	sourceIDCountMap := make(map[string]int)
	for i, entity := range entities {
		entityNames[i] = entity.Name

		// Initially collect all source IDs from primary entities
		arrSourceID := strings.SplitSeq(entity.SourceIDs, GraphFieldSeparator)
		for sourceID := range arrSourceID {
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

	// Get related entities to find their sources too
	relatedEntitiesMap, err := storage.GraphRelatedEntities(entityNames)
	if err != nil {
		return nil, fmt.Errorf("failed to get related entities: %w", err)
	}

	// Count occurrences of each source in related entities to determine relevance
	for _, relEntities := range relatedEntitiesMap {
		for _, relEntity := range relEntities {
			arrSourceID := strings.SplitSeq(relEntity.SourceIDs, GraphFieldSeparator)
			for sourceID := range arrSourceID {
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

	// Retrieve actual source content for each ID and build the result
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
	// Extract all unique entity names from both sides of the relationships
	entityNames := make([]string, 0, len(relationships))
	for _, rel := range relationships {
		entityNames = appendIfUnique(entityNames, rel.SourceEntity)
		entityNames = appendIfUnique(entityNames, rel.TargetEntity)
	}

	// Get full entity details from storage
	entitiesMap, err := storage.GraphEntities(entityNames)
	if err != nil {
		return nil, fmt.Errorf("failed to batch get entities: %w", err)
	}

	// Get relationship counts to determine entity importance
	refCountMap, err := storage.GraphCountEntitiesRelationships(entityNames)
	if err != nil {
		return nil, fmt.Errorf("failed to batch count relationships: %w", err)
	}

	// Construct result with reference counts for ranking
	entities := make([]EntityContext, 0, len(entitiesMap))
	for name, entity := range entitiesMap {
		refCount, ok := refCountMap[name]
		if !ok {
			refCount = 0
		}
		entities = append(entities, EntityContext{
			Name:        entity.Name,
			Type:        entity.Type,
			Description: entity.Descriptions,
			RefCount:    refCount,
			CreatedAt:   entity.CreatedAt,
		})
	}

	return entities, nil
}

func relationshipsRankedSources(relationships []GraphRelationship, storage Storage) ([]SourceContext, error) {
	// Track sources and their reference counts across relationships
	sourcesMap := make(map[string]SourceContext)
	for _, rel := range relationships {
		// Parse source IDs from the relationship
		arrSourceIDs := strings.SplitSeq(rel.SourceIDs, GraphFieldSeparator)
		for sourceID := range arrSourceIDs {
			if sourceID == "" {
				continue
			}

			// Retrieve source content from storage
			source, err := storage.KVSource(sourceID)
			if err != nil {
				return nil, fmt.Errorf("failed to get source with id %s: %w", sourceID, err)
			}

			// Initialize source entry if it's new
			_, ok := sourcesMap[sourceID]
			if !ok {
				sourcesMap[sourceID] = SourceContext{
					Content: source.Content,
				}
			}

			// Increment reference count for this source
			src := sourcesMap[sourceID]
			src.RefCount++
			sourcesMap[sourceID] = src
		}
	}

	// Convert map to slice for return
	sources := make([]SourceContext, len(sourcesMap))
	i := 0
	for _, source := range sourcesMap {
		sources[i] = source
		i++
	}

	return sources, nil
}

func combineContexts(headers []string, ctx1, ctx2 []refContext) string {
	// Merge contexts from both sources, with later ones overwriting duplicates
	resMap := make(map[string]refContext)
	for _, ctx := range ctx1 {
		resMap[ctx.context] = ctx
	}
	for _, ctx := range ctx2 {
		resMap[ctx.context] = ctx
	}

	// Convert map to slice for sorting
	arrRes := make([]refContext, 0, len(resMap))
	for _, ctx := range resMap {
		arrRes = append(arrRes, ctx)
	}

	// Sort by reference count in descending order (most relevant first)
	slices.SortFunc(arrRes, func(a, b refContext) int {
		return cmp.Compare(b.refCount, a.refCount)
	})

	// Format as CSV with numbered IDs
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
