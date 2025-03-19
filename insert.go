package golightrag

import (
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/MegaGrindStone/go-light-rag/internal"
	"golang.org/x/sync/errgroup"
)

// DocumentHandler provides an interface for processing documents and interacting with language models.
type DocumentHandler interface {
	ChunksDocument(content string) ([]Source, error)
	// EntityExtractionPromptData returns the data needed to generate prompts for extracting
	// entities and relationships from text content.
	// The implementation doesn't need to fill the Input field, as it will be filled in the
	// Insert function.
	EntityExtractionPromptData() EntityExtractionPromptData
	// MaxRetries determines the maximum number of retries allowed for the Chat function.
	// This is especially used when extracting entities and relationships from text content,
	// due to the incorrect format that sometimes LLM returns.
	MaxRetries() int
	// ConcurrencyCount determines the number of concurrent requests to the LLM.
	ConcurrencyCount() int
	// BackoffDuration determines the backoff duration between retries.
	BackoffDuration() time.Duration
	GleanCount() int
	MaxSummariesTokenLength() int
}

// Document represents a text document to be processed and stored.
// It contains an ID for unique identification and the content to be analyzed.
type Document struct {
	ID      string
	Content string
}

type summarizeDescriptionsPromptData struct {
	EntityName   string
	Descriptions string
	Language     string
}

// GraphFieldSeparator is a constant used to separate fields in a graph.
const GraphFieldSeparator = "<SEP>"

// Insert processes a document and stores it in the provided storage.
// It chunks the document content, extracts entities and relationships using the provided
// document handler, and stores the results in the appropriate storage.
// It returns an error if any step in the process fails.
func Insert(doc Document, handler DocumentHandler, storage Storage, llm LLM, logger *slog.Logger) error {
	content := cleanContent(doc.Content)

	logger = logger.With(
		slog.String("package", "golightrag"),
		slog.String("function", "Insert"),
	)

	chunks, err := handler.ChunksDocument(content)
	if err != nil {
		return fmt.Errorf("failed to chunk string: %w", err)
	}

	// The chunks returned from the ChunksDocument doesn't have an ID, generate one here
	// based on the document ID and the order of the chunks. This ID would be used to retrieve
	// the chunk in the Query function.
	chunksWithID := make([]Source, len(chunks))
	for i, chunk := range chunks {
		id := chunk.genID(doc.ID)
		chunksWithID[i] = Source{
			ID:         id,
			Content:    chunk.Content,
			TokenSize:  chunk.TokenSize,
			OrderIndex: chunk.OrderIndex,
		}
	}

	logger.Info("Upserting sources", "count", len(chunks))

	if err := storage.KVUpsertSources(chunksWithID); err != nil {
		return fmt.Errorf("failed to upsert sources kv: %w", err)
	}

	llmConcurrencyCount := handler.ConcurrencyCount()
	if llmConcurrencyCount == 0 {
		llmConcurrencyCount = 1
	}

	if err := extractEntities(doc.ID, chunks, llm,
		handler.EntityExtractionPromptData(), handler.MaxRetries(), llmConcurrencyCount, handler.GleanCount(),
		handler.MaxSummariesTokenLength(), handler.BackoffDuration(), storage, logger); err != nil {
		return fmt.Errorf("failed to extract entities: %w", err)
	}

	return nil
}

func extractEntities(
	docID string,
	sources []Source,
	llm LLM,
	extractPromptData EntityExtractionPromptData,
	llmMaxRetries, llmConcurrencyCount, llmMaxGleanCount, summariesMaxToken int,
	backoffDuration time.Duration,
	storage Storage,
	logger *slog.Logger,
) error {
	orderedSources := make([]Source, len(sources))
	copy(orderedSources, sources)
	sort.Slice(orderedSources, func(i, j int) bool {
		return orderedSources[i].OrderIndex < orderedSources[j].OrderIndex
	})

	logger.Info("Extracting entities", "count", len(orderedSources))

	eg := new(errgroup.Group)
	sem := make(chan struct{}, llmConcurrencyCount)

	for i, source := range orderedSources {
		eg.Go(func() error {
			sem <- struct{}{}
			defer func() { <-sem }()

			entities, relationships, err := llmExtractEntities(source.Content,
				extractPromptData, llmMaxRetries, llmMaxGleanCount, backoffDuration, llm, logger)
			if err != nil {
				return fmt.Errorf("failed to extract entities with LLM: %w", err)
			}

			logger.Info("Done call LLM", "entities", len(entities), "relationships", len(relationships))

			for name, unmergedEntities := range entities {
				if err := mergeGraphEntities(name, source.genID(docID), extractPromptData.Language,
					unmergedEntities, summariesMaxToken, storage, llm, logger); err != nil {
					return fmt.Errorf("failed to process graph entity: %w", err)
				}
			}

			for key, unmergedRelationships := range relationships {
				if err := mergeGraphRelationships(key, source.genID(docID), extractPromptData.Language,
					unmergedRelationships, summariesMaxToken, storage, llm, logger); err != nil {
					return fmt.Errorf("failed to process graph relationship: %w", err)
				}
			}

			logger.Info("Processed source", "index", i+1)

			return nil
		})
	}

	if err := eg.Wait(); err != nil {
		return err
	}

	return nil
}

func llmExtractEntities(
	content string,
	data EntityExtractionPromptData,
	maxRetries, maxGleanCount int,
	backoffDuration time.Duration,
	llm LLM,
	logger *slog.Logger,
) (map[string][]GraphEntity, map[string][]GraphRelationship, error) {
	data.Input = content
	extractPrompt, err := promptTemplate("extract-entities", extractEntitiesPrompt, data)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate extract entities prompt: %w", err)
	}
	gleanPrompt, err := promptTemplate("glean-entities", gleanEntitiesPrompt, data)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to generate glean entities prompt: %w", err)
	}

	logger.Debug("Use LLM to extract entities from source",
		"extractPrompt", extractPrompt, "gleanPrompt", gleanPrompt, "source", content)

	type llmResult struct {
		Entities      []GraphEntity       `json:"entities"`
		Relationships []GraphRelationship `json:"relationships"`
	}

	var results llmResult

	retry := 0

	for {
		// If this is not a first retry, add backoff delay.
		if retry > 0 {
			time.Sleep(backoffDuration)
		}
		// LLM sometimes returns incorrect format, retry up to maxRetries() times.
		if retry >= maxRetries {
			return nil, nil, fmt.Errorf("failed to extract entities after %d retries", maxRetries)
		}

		logger.Debug("Use LLM to extract entities from source", "extractPrompt", extractPrompt)

		histories := []string{extractPrompt}

		sourceResult, err := llm.Chat(histories)
		if err != nil {
			nErr := fmt.Errorf("failed to call LLM: %w", err)
			retry++
			logger.Warn("Retry extract", "retry", retry, "error", nErr)
			continue
		}

		var sourceParsed llmResult
		err = json.Unmarshal([]byte(sourceResult), &sourceParsed)
		if err != nil {
			nErr := fmt.Errorf("failed to parse llm result: %w", err)
			retry++
			logger.Warn("Retry parse result", "retry", retry, "error", nErr)
			continue
		}
		results.Entities = append(results.Entities, sourceParsed.Entities...)
		results.Relationships = append(results.Relationships, sourceParsed.Relationships...)

		histories = append(histories, sourceResult)

		gleanCount := 0
		for {
			logger.Debug("Use LLM to glean entities from source", "gleanPrompt", gleanPrompt)
			histories = append(histories, gleanPrompt)
			gleanResult, err := llm.Chat(histories)
			if err != nil {
				nErr := fmt.Errorf("failed to call LLM on glean: %w", err)
				retry++
				logger.Warn("Retry glean", "retry", retry, "error", nErr)
				continue
			}

			histories = append(histories, gleanResult)

			var gleanParsed llmResult
			err = json.Unmarshal([]byte(gleanResult), &gleanParsed)
			if err != nil {
				nErr := fmt.Errorf("failed to parse llm result: %w", err)
				retry++
				logger.Warn("Retry parse result", "retry", retry, "error", nErr)
				continue
			}
			results.Entities = append(results.Entities, gleanParsed.Entities...)
			results.Relationships = append(results.Relationships, gleanParsed.Relationships...)

			gleanCount++
			if gleanCount > maxGleanCount {
				break
			}

			decideMessages := make([]string, 0)
			decideMessages = append(decideMessages, histories...)
			decideMessages = append(decideMessages, gleanDecideContinuePrompt)

			decideResult, err := llm.Chat(decideMessages)
			if err != nil {
				nErr := fmt.Errorf("failed to call LLM on decide: %w", err)
				retry++
				logger.Warn("Retry decide", "retry", retry, "error", nErr)
				continue
			}

			decideResult = strings.ToLower(strings.TrimSpace(strings.Trim(strings.Trim(decideResult, `"`), `'`)))

			logger.Debug("Decide result from LLM", "decideResult", decideResult)

			if decideResult != "yes" {
				break
			}
		}

		entities, relationships := dedupeLLMResult(results.Entities, results.Relationships, data.EntityTypes)
		return entities, relationships, nil
	}
}

func dedupeLLMResult(
	entities []GraphEntity,
	relationships []GraphRelationship,
	entityTypes []string,
) (map[string][]GraphEntity, map[string][]GraphRelationship) {
	ents := make(map[string][]GraphEntity, 0)
	rels := make(map[string][]GraphRelationship, 0)

	expectedEntityTypes := make([]string, 0)
	for _, et := range entityTypes {
		expectedEntityTypes = append(expectedEntityTypes, strings.ToUpper(et))
	}
	expectedEntityTypes = append(expectedEntityTypes, "UNKNOWN")

	for _, entity := range entities {
		entity.Type = strings.ToUpper(entity.Type)
		if !slices.Contains(expectedEntityTypes, entity.Type) {
			// If llm returns invalid entity type, use UNKNOWN.
			entity.Type = "UNKNOWN"
		}

		entity.Name = strings.ToUpper(entity.Name)
		if _, ok := ents[entity.Name]; !ok {
			ents[entity.Name] = make([]GraphEntity, 0)
		}
		ents[entity.Name] = append(ents[entity.Name], entity)
	}

	for _, relationship := range relationships {
		relationship.SourceEntity = strings.ToUpper(relationship.SourceEntity)
		relationship.TargetEntity = strings.ToUpper(relationship.TargetEntity)
		relationKey := fmt.Sprintf("%s-%s", relationship.SourceEntity, relationship.TargetEntity)
		if _, ok := rels[relationKey]; !ok {
			rels[relationKey] = make([]GraphRelationship, 0)
		}
		rels[relationKey] = append(rels[relationKey], relationship)
	}

	return ents, rels
}

func mergeGraphEntities(
	name, sourceID, language string,
	entities []GraphEntity,
	summariesMaxToken int,
	storage Storage,
	llm LLM,
	logger *slog.Logger,
) error {
	existingTypes := make([]string, 0)
	existingSourceIDs := make([]string, 0)
	existingDescriptions := make([]string, 0)

	existingEntity, err := storage.GraphEntity(name)
	if err != nil {
		if !errors.Is(err, ErrEntityNotFound) {
			return fmt.Errorf("failed to get entity: %w", err)
		}
	} else {
		existingTypes = append(existingTypes, existingEntity.Type)

		arrDescriptions := strings.Split(existingEntity.Descriptions, GraphFieldSeparator)
		existingDescriptions = append(existingDescriptions, arrDescriptions...)

		arrSourceIDs := strings.Split(existingEntity.SourceIDs, GraphFieldSeparator)
		existingSourceIDs = append(existingSourceIDs, arrSourceIDs...)
	}

	for _, entity := range entities {
		existingTypes = append(existingTypes, entity.Type)
		existingDescriptions = appendIfUnique(existingDescriptions, entity.Descriptions)
	}
	existingSourceIDs = appendIfUnique(existingSourceIDs, sourceID)

	entityType := mostFrequentItem(existingTypes)
	sourceIDs := strings.Join(existingSourceIDs, GraphFieldSeparator)
	description, err := descriptionsSummary(name, language, summariesMaxToken, existingDescriptions, llm)
	if err != nil {
		return fmt.Errorf("failed to summarize descriptions: %w", err)
	}

	ent := GraphEntity{
		Name:         name,
		Type:         entityType,
		Descriptions: description,
		SourceIDs:    sourceIDs,
		CreatedAt:    time.Now(),
	}

	logger.Debug("Upserting graph entity", "entity", ent)

	if err := storage.GraphUpsertEntity(ent); err != nil {
		return fmt.Errorf("failed to upsert graph entity in graph storage: %w", err)
	}

	if err := storage.VectorUpsertEntity(ent.Name, ent.Name+ent.Descriptions); err != nil {
		return fmt.Errorf("failed to upsert entity in vector storage: %w", err)
	}

	return nil
}

func mergeGraphRelationships(
	key, sourceID, language string,
	relationships []GraphRelationship,
	summariesMaxToken int,
	storage Storage,
	llm LLM,
	logger *slog.Logger,
) error {
	existingWeight := 0.0
	existingDescriptions := make([]string, 0)
	existingKeywords := make([]string, 0)
	existingSourceIDs := make([]string, 0)

	arrKey := strings.Split(key, "-")
	sourceEntity := arrKey[0]
	targetEntity := arrKey[1]

	existingRelationship, err := storage.GraphRelationship(sourceEntity, targetEntity)
	if err != nil {
		if !errors.Is(err, ErrRelationshipNotFound) {
			return fmt.Errorf("failed to get relationship: %w", err)
		}
	} else {
		existingWeight += existingRelationship.Weight

		arrDescriptions := strings.Split(existingRelationship.Descriptions, GraphFieldSeparator)
		existingDescriptions = append(existingDescriptions, arrDescriptions...)

		existingKeywords = append(existingKeywords, existingRelationship.Keywords...)

		arrSourceIDs := strings.Split(existingRelationship.SourceIDs, GraphFieldSeparator)
		existingSourceIDs = append(existingSourceIDs, arrSourceIDs...)
	}

	for _, relationship := range relationships {
		existingWeight += relationship.Weight
		existingDescriptions = appendIfUnique(existingDescriptions, relationship.Descriptions)
		for _, keyword := range relationship.Keywords {
			existingKeywords = appendIfUnique(existingKeywords, keyword)
		}
	}
	existingSourceIDs = appendIfUnique(existingSourceIDs, sourceID)

	description, err := descriptionsSummary(key, language, summariesMaxToken, existingDescriptions, llm)
	if err != nil {
		return fmt.Errorf("failed to summarize descriptions: %w", err)
	}
	sourceIDs := strings.Join(existingSourceIDs, GraphFieldSeparator)

	_, err = storage.GraphEntity(sourceEntity)
	if err != nil {
		if !errors.Is(err, ErrEntityNotFound) {
			return fmt.Errorf("failed to get source entity with name %s: %w", sourceEntity, err)
		}
		logger.Debug("Entity not found, upserting", "entity", sourceEntity)

		if err := storage.GraphUpsertEntity(GraphEntity{
			Name:         sourceEntity,
			Type:         "UNKNOWN",
			Descriptions: description,
			SourceIDs:    sourceID,
			CreatedAt:    time.Now(),
		}); err != nil {
			return fmt.Errorf("failed to upsert source node with name %s: %w", sourceEntity, err)
		}
	}

	_, err = storage.GraphEntity(targetEntity)
	if err != nil {
		if !errors.Is(err, ErrEntityNotFound) {
			return fmt.Errorf("failed to get target entity with name %s: %w", targetEntity, err)
		}
		logger.Debug("Entity not found, upserting", "entity", targetEntity)
		if err := storage.GraphUpsertEntity(GraphEntity{
			Name:         targetEntity,
			Type:         "UNKNOWN",
			Descriptions: description,
			SourceIDs:    sourceID,
			CreatedAt:    time.Now(),
		}); err != nil {
			return fmt.Errorf("failed to upsert target node with name %s: %w", targetEntity, err)
		}
	}

	rel := GraphRelationship{
		SourceEntity: sourceEntity,
		TargetEntity: targetEntity,
		Weight:       existingWeight,
		Descriptions: description,
		Keywords:     existingKeywords,
		SourceIDs:    sourceIDs,
		CreatedAt:    time.Now(),
	}

	if err := storage.GraphUpsertRelationship(rel); err != nil {
		return fmt.Errorf("failed to upsert graph relationship: %w", err)
	}

	keywords := strings.Join(rel.Keywords, GraphFieldSeparator)
	content := keywords + rel.SourceEntity + rel.TargetEntity + rel.Descriptions
	if err := storage.VectorUpsertRelationship(rel.SourceEntity, rel.TargetEntity, content); err != nil {
		return fmt.Errorf("failed to upsert relationship vector: %w", err)
	}

	return nil
}

func descriptionsSummary(name, language string, maxToken int, descriptions []string, llm LLM) (string, error) {
	joinedDescriptions := strings.Join(descriptions, GraphFieldSeparator)
	tokens, err := internal.EncodeStringByTiktoken(joinedDescriptions)
	if err != nil {
		return "", fmt.Errorf("failed to encode string: %w", err)
	}
	if len(tokens) < maxToken {
		return joinedDescriptions, nil
	}
	descString := strings.Join(descriptions, ", ")
	descString = "[" + descString + "]"

	summarizePrompt, err := promptTemplate("summarize-descriptions", summarizeDescriptionsPrompt,
		summarizeDescriptionsPromptData{
			EntityName:   name,
			Descriptions: descString,
			Language:     language,
		})
	if err != nil {
		return "", fmt.Errorf("failed to generate summarize descriptions prompt: %w", err)
	}

	return llm.Chat([]string{summarizePrompt})
}
