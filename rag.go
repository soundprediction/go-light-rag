package golightrag

import (
	"errors"
	"fmt"
	"strings"
	"text/template"
	"time"
)

type LLM interface {
	Chat(messages []string) (string, error)
	MaxRetries() int
	GleanCount() int
	MaxSummariesTokenLength() int
}

type GraphStorage interface {
	GraphEntity(name string) (GraphEntity, error)
	GraphRelationship(sourceEntity, targetEntity string) (GraphRelationship, error)

	GraphUpsertEntity(entity GraphEntity) error
	GraphUpsertRelationship(relationship GraphRelationship) error

	GraphCountEntityRelationships(name string) (int, error)
	GraphRelatedEntities(name string) ([]GraphEntity, error)
}

type VectorStorage interface {
	VectorQueryEntity(keywords string) ([]string, error)
	VectorQueryRelationship(keywords string) ([][2]string, error)

	VectorUpsertEntity(name, content string) error
	VectorUpsertRelationship(source, target, content string) error
}

type KeyValueStorage interface {
	KVSource(id string) (Source, error)
	KVUpsertSources(sources []Source) error
}

type Storage interface {
	GraphStorage
	VectorStorage
	KeyValueStorage
}

type Source struct {
	ID         string
	Content    string
	TokenSize  int
	OrderIndex int
}

type GraphEntity struct {
	Name         string
	Type         string
	Descriptions string
	SourceIDs    string
	CreatedAt    time.Time
}

type GraphRelationship struct {
	SourceEntity string
	TargetEntity string
	Weight       float64
	Descriptions string
	Keywords     string
	SourceIDs    string
	CreatedAt    time.Time
}

var (
	ErrEntityNotFound       = errors.New("entity not found")
	ErrRelationshipNotFound = errors.New("relationship not found")
)

func cleanContent(content string) string {
	// Removes spaces and null characters.
	str := strings.TrimSpace(content)
	return strings.ReplaceAll(str, "\x00", "")
}

func promptTemplate(name, templ string, data any) (string, error) {
	buf := strings.Builder{}
	tmpl := template.New(name).Funcs(template.FuncMap{
		"add": func(a, b int) int {
			return a + b
		},
	})
	tmpl = template.Must(tmpl.Parse(templ))
	if err := tmpl.Execute(&buf, data); err != nil {
		return "", fmt.Errorf("failed to execute template: %w", err)
	}

	return buf.String(), nil
}

func appendIfUnique(slice []string, item string) []string {
	for _, ele := range slice {
		if ele == item {
			return slice
		}
	}
	return append(slice, item)
}

func mostFrequentItem(list []string) string {
	// Create a map to store counts
	counts := make(map[string]int)

	// Count occurrences of each string
	for _, item := range list {
		counts[item]++
	}

	// Find the item with highest count
	maxCount := 0
	var mostFreqItem string

	for item, count := range counts {
		if count > maxCount {
			maxCount = count
			mostFreqItem = item
		}
	}

	return mostFreqItem
}

func threeBacktick(caption string) string {
	return "```" + caption
}

func (s Source) genID(docID string) string {
	return fmt.Sprintf("%s-chunk-%d", docID, s.OrderIndex)
}
