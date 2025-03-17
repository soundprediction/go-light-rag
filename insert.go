package golightrag

import (
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/MegaGrindStone/go-light-rag/internal"
)

// DocumentHandler provides an interface for processing documents and interacting with language models.
// It defines methods for chunking documents, providing entity extraction prompt data,
// and accessing the underlying language model for operations.
type DocumentHandler interface {
	ChunksDocument(content string) ([]Source, error)
	// EntityExtractionPromptData returns the data needed to generate prompts for extracting
	// entities and relationships from text content.
	// The implementation doesn't need to fill the Input field, as it will be filled in the
	// Insert function.
	EntityExtractionPromptData() EntityExtractionPromptData
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

const (
	fieldDelimiter      = "<|>"
	recordDelimiter     = "##"
	completeDelimiter   = "<|COMPLETE|>"
	graphFieldSeparator = "<SEP>"
)

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

	if err := extractEntities(doc.ID, chunks, llm, handler.EntityExtractionPromptData(), storage, logger); err != nil {
		return fmt.Errorf("failed to extract entities: %w", err)
	}

	return nil
}

func extractEntities(
	docID string,
	sources []Source,
	llm LLM,
	extractPromptData EntityExtractionPromptData,
	storage Storage,
	logger *slog.Logger,
) error {
	orderedSources := make([]Source, len(sources))
	copy(orderedSources, sources)
	sort.Slice(orderedSources, func(i, j int) bool {
		return orderedSources[i].OrderIndex < orderedSources[j].OrderIndex
	})

	logger.Info("Extracting entities", "count", len(orderedSources))

	for i, source := range orderedSources {
		// TODO: Call this using concurrency
		entities, relationships, err := llmExtractEntities(source.Content, extractPromptData, llm, logger)
		if err != nil {
			return fmt.Errorf("failed to extract entities with LLM: %w", err)
		}

		logger.Info("Done call LLM", "entities", len(entities), "relationships", len(relationships))

		for name, unmergedEntities := range entities {
			if err := mergeGraphEntities(name, source.genID(docID), extractPromptData.Language,
				unmergedEntities, storage, llm, logger); err != nil {
				return fmt.Errorf("failed to process graph entity: %w", err)
			}
		}

		for key, unmergedRelationships := range relationships {
			if err := mergeGraphRelationships(key, source.genID(docID), extractPromptData.Language,
				unmergedRelationships, storage, llm, logger); err != nil {
				return fmt.Errorf("failed to process graph relationship: %w", err)
			}
		}

		logger.Info("Processed source", "index", i+1)
	}

	return nil
}

func llmExtractEntities(
	content string,
	data EntityExtractionPromptData,
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

	retry := 0

	var entities map[string][]GraphEntity
	var relationships map[string][]GraphRelationship

	for {
		// LLM sometimes returns incorrect format, retry up to llm.MaxRetries() times.
		if retry > llm.MaxRetries() {
			return nil, nil, fmt.Errorf("failed to extract entities after %d retries", llm.MaxRetries())
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
			sourceResult += gleanResult

			gleanCount++
			if gleanCount > llm.GleanCount() {
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

		entities, relationships, err = parseLLMResult(sourceResult)
		if err != nil {
			nErr := fmt.Errorf("failed to parse llm result: %w", err)
			retry++
			logger.Warn("Retry parse result", "retry", retry, "error", nErr)
			continue
		}

		return entities, relationships, nil
	}
}

// parseLLMResult parses the result from LLM and returns a map of entities and relationships
// based on defined at extractEntitiesPrompt. Although it can be done using responseFormat or
// structuredOutput, many LLM models is not supporting it yet.
func parseLLMResult(result string) (map[string][]GraphEntity, map[string][]GraphRelationship, error) {
	arrBatch := strings.Split(result, completeDelimiter)
	if len(arrBatch) <= 1 {
		return nil, nil, fmt.Errorf("no complete delimiter found in result: %s", result)
	}

	entities := make(map[string][]GraphEntity, 0)
	relationships := make(map[string][]GraphRelationship, 0)

	for _, batch := range arrBatch {
		arrRes := strings.Split(batch, recordDelimiter)

		for _, res := range arrRes {
			res = strings.TrimSpace(res)
			res = strings.TrimPrefix(res, "(")
			res = strings.TrimSuffix(res, ")")
			if res == "" {
				continue
			}

			arrFields := strings.Split(res, fieldDelimiter)
			if len(arrFields) < 1 {
				return nil, nil, fmt.Errorf("no fields found in result")
			}

			field := arrFields[0]
			field = strings.Trim(field, "\"")

			switch field {
			case "entity":
				if len(arrFields) != 4 {
					return nil, nil, fmt.Errorf("invalid entity format from result: %s", res)
				}
				entityName := strings.ToUpper(normalizeLLMResultField(arrFields[1]))
				if entityName == "" {
					continue
				}
				entityType := strings.ToUpper(normalizeLLMResultField(arrFields[2]))
				if entityType == "" {
					entityType = "UNKNOWN"
				}

				entity := GraphEntity{
					Name:         entityName,
					Type:         entityType,
					Descriptions: normalizeLLMResultField(arrFields[3]),
				}
				if _, ok := entities[entity.Name]; !ok {
					entities[entity.Name] = make([]GraphEntity, 0)
				}
				entities[entity.Name] = append(entities[entity.Name], entity)
			case "relationship":
				if len(arrFields) != 6 {
					return nil, nil, fmt.Errorf("invalid relationship format from result: %s", res)
				}
				sourceEntity := strings.ToUpper(normalizeLLMResultField(arrFields[1]))
				if sourceEntity == "" {
					continue
				}
				targetEntity := strings.ToUpper(normalizeLLMResultField(arrFields[2]))
				if targetEntity == "" {
					continue
				}

				weight, err := strconv.ParseFloat(arrFields[5], 64)
				if err != nil {
					weight = 1.0
				}
				relationship := GraphRelationship{
					SourceEntity: sourceEntity,
					TargetEntity: targetEntity,
					Descriptions: normalizeLLMResultField(arrFields[3]),
					Keywords:     normalizeLLMResultField(arrFields[4]),
					Weight:       weight,
				}
				relationKey := fmt.Sprintf("%s-%s", relationship.SourceEntity, relationship.TargetEntity)
				if _, ok := relationships[relationKey]; !ok {
					relationships[relationKey] = make([]GraphRelationship, 0)
				}
				relationships[relationKey] = append(relationships[relationKey], relationship)
			case "content_keywords":
				// Unused fields
				continue
			default:
				return nil, nil, fmt.Errorf("invalid field type: %s, from result: %s", field, res)
			}
		}
	}

	return entities, relationships, nil
}

func normalizeLLMResultField(s string) string {
	// Trim whitespace
	result := strings.TrimSpace(s)

	// Trim leading and trailing double quotes
	result = strings.TrimPrefix(result, "\"")
	result = strings.TrimSuffix(result, "\"")

	// Remove control characters in ranges 0x00-0x1F and 0x7F-0x9F
	re := regexp.MustCompile(`[\x00-\x1f\x7f-\x9f]`)
	return re.ReplaceAllString(result, "")
}

func mergeGraphEntities(
	name, sourceID, language string,
	entities []GraphEntity,
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

		arrDescriptions := strings.Split(existingEntity.Descriptions, graphFieldSeparator)
		existingDescriptions = append(existingDescriptions, arrDescriptions...)

		arrSourceIDs := strings.Split(existingEntity.SourceIDs, graphFieldSeparator)
		existingSourceIDs = append(existingSourceIDs, arrSourceIDs...)
	}

	for _, entity := range entities {
		existingTypes = append(existingTypes, entity.Type)
		existingDescriptions = appendIfUnique(existingDescriptions, entity.Descriptions)
	}
	existingSourceIDs = appendIfUnique(existingSourceIDs, sourceID)

	entityType := mostFrequentItem(existingTypes)
	sourceIDs := strings.Join(existingSourceIDs, graphFieldSeparator)
	description, err := descriptionsSummary(name, language, existingDescriptions, llm)
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

		arrDescriptions := strings.Split(existingRelationship.Descriptions, graphFieldSeparator)
		existingDescriptions = append(existingDescriptions, arrDescriptions...)

		arrKeywords := strings.Split(existingRelationship.Keywords, graphFieldSeparator)
		existingKeywords = append(existingKeywords, arrKeywords...)

		arrSourceIDs := strings.Split(existingRelationship.SourceIDs, graphFieldSeparator)
		existingSourceIDs = append(existingSourceIDs, arrSourceIDs...)
	}

	for _, relationship := range relationships {
		existingWeight += relationship.Weight
		existingDescriptions = appendIfUnique(existingDescriptions, relationship.Descriptions)
		existingKeywords = appendIfUnique(existingKeywords, relationship.Keywords)
	}
	existingSourceIDs = appendIfUnique(existingSourceIDs, sourceID)

	description, err := descriptionsSummary(key, language, existingDescriptions, llm)
	if err != nil {
		return fmt.Errorf("failed to summarize descriptions: %w", err)
	}
	keywords := strings.Join(existingKeywords, graphFieldSeparator)
	sourceIDs := strings.Join(existingSourceIDs, graphFieldSeparator)

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
		Keywords:     keywords,
		SourceIDs:    sourceIDs,
		CreatedAt:    time.Now(),
	}

	if err := storage.GraphUpsertRelationship(rel); err != nil {
		return fmt.Errorf("failed to upsert graph relationship: %w", err)
	}

	content := rel.Keywords + rel.SourceEntity + rel.TargetEntity + rel.Descriptions
	if err := storage.VectorUpsertRelationship(rel.SourceEntity, rel.TargetEntity, content); err != nil {
		return fmt.Errorf("failed to upsert relationship vector: %w", err)
	}

	return nil
}

func descriptionsSummary(name, language string, descriptions []string, llm LLM) (string, error) {
	joinedDescriptions := strings.Join(descriptions, graphFieldSeparator)
	tokens, err := internal.EncodeStringByTiktoken(joinedDescriptions)
	if err != nil {
		return "", fmt.Errorf("failed to encode string: %w", err)
	}
	if len(tokens) < llm.MaxSummariesTokenLength() {
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
