package handler

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

	golightrag "github.com/MegaGrindStone/go-light-rag"
	"github.com/MegaGrindStone/go-light-rag/internal"
)

// ChunkingOptions defines configuration for text chunking
type ChunkingOptions struct {
	MaxChunkSize         int     // Maximum characters per chunk
	MinChunkSize         int     // Minimum characters per chunk
	OverlapSize          int     // Number of characters to overlap between chunks
	SentenceWeight       float64 // Weight for sentence boundaries (0-1)
	ParagraphWeight      float64 // Weight for paragraph boundaries (0-1)
	HeadingWeight        float64 // Weight for heading boundaries (0-1)
	CodeBlockWeight      float64 // Weight for code block boundaries (0-1)
	ListWeight           float64 // Weight for list boundaries (0-1)
	TableWeight          float64 // Weight for table boundaries (0-1)
	BlockquoteWeight     float64 // Weight for blockquote boundaries (0-1)
	HorizontalRuleWeight float64 // Weight for horizontal rule boundaries (0-1)
	PreserveFormatting   bool    // Whether to preserve whitespace formatting
	RespectCodeBlocks    bool    // Never split inside code blocks
	RespectTables        bool    // Never split inside tables
	HeaderHierarchy      bool    // Consider heading levels in chunking decisions
}

// DefaultMarkdownChunkingOptions returns sensible defaults for Markdown
func DefaultMarkdownChunkingOptions() ChunkingOptions {
	return ChunkingOptions{
		MaxChunkSize:         1500,
		MinChunkSize:         200,
		OverlapSize:          50,
		SentenceWeight:       0.3,
		ParagraphWeight:      0.5,
		HeadingWeight:        1.0,
		CodeBlockWeight:      0.9,
		ListWeight:           0.4,
		TableWeight:          0.8,
		BlockquoteWeight:     0.6,
		HorizontalRuleWeight: 0.8,
		PreserveFormatting:   true,
		RespectCodeBlocks:    true,
		RespectTables:        true,
		HeaderHierarchy:      true,
	}
}

// Chunk represents a text chunk with metadata
type Chunk struct {
	Text         string
	StartPos     int
	EndPos       int
	ChunkType    string                 // "heading", "code_block", "table", "list", "paragraph", "sentence", "arbitrary"
	Score        float64                // Semantic boundary score
	HeadingLevel int                    // For heading chunks, the level (1-6)
	Metadata     map[string]interface{} // Additional metadata
}

// BoundaryMarker represents a potential chunking point
type BoundaryMarker struct {
	Position     int
	Type         string
	Score        float64
	Context      string
	HeadingLevel int
	Metadata     map[string]interface{}
}

// MarkdownElement represents a parsed markdown element
type MarkdownElement struct {
	Type     string // "heading", "code_block", "table", "list", "blockquote", "paragraph", etc.
	StartPos int
	EndPos   int
	Level    int    // For headings, the level (1-6)
	Language string // For code blocks, the language
	Content  string
	Metadata map[string]interface{}
}

// SemanticChunker handles text chunking operations with Markdown awareness
type SemanticChunker struct {
	options ChunkingOptions

	// Compiled regex patterns for Markdown parsing
	headingPattern        *regexp.Regexp
	codeBlockPattern      *regexp.Regexp
	inlineCodePattern     *regexp.Regexp
	tablePattern          *regexp.Regexp
	listItemPattern       *regexp.Regexp
	blockquotePattern     *regexp.Regexp
	horizontalRulePattern *regexp.Regexp
	linkPattern           *regexp.Regexp
	imagePattern          *regexp.Regexp
	sentencePattern       *regexp.Regexp
	paragraphPattern      *regexp.Regexp
}

// NewMarkdownChunker creates a new chunker optimized for Markdown
func NewMarkdownChunker(options ChunkingOptions) *SemanticChunker {
	return &SemanticChunker{
		options: options,

		// Markdown-specific patterns
		headingPattern:        regexp.MustCompile(`(?m)^(#{1,6})\s+(.+)$`),
		codeBlockPattern:      regexp.MustCompile("(?s)```(\\w+)?\\n?(.*?)```"),
		inlineCodePattern:     regexp.MustCompile("`[^`]+`"),
		tablePattern:          regexp.MustCompile(`(?m)^\|.*\|$`),
		listItemPattern:       regexp.MustCompile(`(?m)^(\s*)([-*+]|\d+\.)\s+(.+)$`),
		blockquotePattern:     regexp.MustCompile(`(?m)^>\s*(.*)$`),
		horizontalRulePattern: regexp.MustCompile(`(?m)^(\*{3,}|-{3,}|_{3,})\s*$`),
		linkPattern:           regexp.MustCompile(`\[([^\]]+)\]\(([^)]+)\)`),
		imagePattern:          regexp.MustCompile(`!\[([^\]]*)\]\(([^)]+)\)`),

		// Text-level patterns
		sentencePattern:  regexp.MustCompile(`[.!?]+\s+`),
		paragraphPattern: regexp.MustCompile(`\n\s*\n`),
	}
}

// ChunkMarkdown performs semantic chunking on Markdown text
func (sc *SemanticChunker) ChunkMarkdown(text string) []Chunk {
	if len(text) <= sc.options.MaxChunkSize {
		return []Chunk{{
			Text:      text,
			StartPos:  0,
			EndPos:    len(text),
			ChunkType: "complete",
			Score:     1.0,
			Metadata:  make(map[string]interface{}),
		}}
	}

	// Parse Markdown structure
	elements := sc.parseMarkdownStructure(text)

	// Find all potential boundary markers
	boundaries := sc.findMarkdownBoundaries(text, elements)

	// Sort boundaries by position
	sort.Slice(boundaries, func(i, j int) bool {
		return boundaries[i].Position < boundaries[j].Position
	})

	// Generate chunks based on boundaries and constraints
	return sc.generateMarkdownChunks(text, boundaries, elements)
}

// parseMarkdownStructure identifies and parses Markdown elements
func (sc *SemanticChunker) parseMarkdownStructure(text string) []MarkdownElement {
	var elements []MarkdownElement

	// Parse headings
	headingMatches := sc.headingPattern.FindAllStringSubmatch(text, -1)
	headingIndices := sc.headingPattern.FindAllStringIndex(text, -1)
	for i, match := range headingMatches {
		if i < len(headingIndices) {
			level := len(match[1]) // Number of # characters
			elements = append(elements, MarkdownElement{
				Type:     "heading",
				StartPos: headingIndices[i][0],
				EndPos:   headingIndices[i][1],
				Level:    level,
				Content:  match[2],
				Metadata: map[string]interface{}{
					"raw_heading": match[0],
					"level":       level,
				},
			})
		}
	}

	// Parse code blocks
	codeBlockMatches := sc.codeBlockPattern.FindAllStringSubmatch(text, -1)
	codeBlockIndices := sc.codeBlockPattern.FindAllStringIndex(text, -1)
	for i, match := range codeBlockMatches {
		if i < len(codeBlockIndices) {
			language := match[1]
			if language == "" {
				language = "text"
			}
			elements = append(elements, MarkdownElement{
				Type:     "code_block",
				StartPos: codeBlockIndices[i][0],
				EndPos:   codeBlockIndices[i][1],
				Language: language,
				Content:  match[2],
				Metadata: map[string]interface{}{
					"language": language,
				},
			})
		}
	}

	// Parse tables
	sc.parseTableElements(text, &elements)

	// Parse lists
	sc.parseListElements(text, &elements)

	// Parse blockquotes
	sc.parseBlockquoteElements(text, &elements)

	// Parse horizontal rules
	horizontalRuleIndices := sc.horizontalRulePattern.FindAllStringIndex(text, -1)
	for _, indices := range horizontalRuleIndices {
		elements = append(elements, MarkdownElement{
			Type:     "horizontal_rule",
			StartPos: indices[0],
			EndPos:   indices[1],
			Content:  text[indices[0]:indices[1]],
			Metadata: make(map[string]interface{}),
		})
	}

	// Sort elements by position
	sort.Slice(elements, func(i, j int) bool {
		return elements[i].StartPos < elements[j].StartPos
	})

	return elements
}

// parseTableElements identifies table structures
func (sc *SemanticChunker) parseTableElements(text string, elements *[]MarkdownElement) {
	lines := strings.Split(text, "\n")
	var tableStart, tableEnd int
	inTable := false

	for i, line := range lines {
		isTableRow := sc.tablePattern.MatchString(line)

		if isTableRow && !inTable {
			// Start of table
			inTable = true
			tableStart = sc.getLineStart(text, i)
		} else if !isTableRow && inTable {
			// End of table
			tableEnd = sc.getLineStart(text, i)
			*elements = append(*elements, MarkdownElement{
				Type:     "table",
				StartPos: tableStart,
				EndPos:   tableEnd,
				Content:  text[tableStart:tableEnd],
				Metadata: map[string]interface{}{
					"row_count": i - sc.getLineNumber(text, tableStart),
				},
			})
			inTable = false
		}
	}

	// Handle table at end of document
	if inTable {
		*elements = append(*elements, MarkdownElement{
			Type:     "table",
			StartPos: tableStart,
			EndPos:   len(text),
			Content:  text[tableStart:],
			Metadata: make(map[string]interface{}),
		})
	}
}

// parseListElements identifies list structures
func (sc *SemanticChunker) parseListElements(text string, elements *[]MarkdownElement) {
	lines := strings.Split(text, "\n")
	var listStart, listEnd int
	inList := false
	currentIndent := -1

	for i, line := range lines {
		matches := sc.listItemPattern.FindStringSubmatch(line)
		isListItem := len(matches) > 0

		if isListItem {
			indent := len(matches[1])

			if !inList || (currentIndent >= 0 && indent <= currentIndent) {
				if inList {
					// End previous list
					listEnd = sc.getLineStart(text, i)
					*elements = append(*elements, MarkdownElement{
						Type:     "list",
						StartPos: listStart,
						EndPos:   listEnd,
						Content:  text[listStart:listEnd],
						Metadata: make(map[string]interface{}),
					})
				}
				// Start new list
				inList = true
				listStart = sc.getLineStart(text, i)
				currentIndent = indent
			}
		} else if inList && strings.TrimSpace(line) == "" {
			// Empty line in list, continue
			continue
		} else if inList {
			// End of list
			listEnd = sc.getLineStart(text, i)
			*elements = append(*elements, MarkdownElement{
				Type:     "list",
				StartPos: listStart,
				EndPos:   listEnd,
				Content:  text[listStart:listEnd],
				Metadata: make(map[string]interface{}),
			})
			inList = false
			currentIndent = -1
		}
	}

	// Handle list at end of document
	if inList {
		*elements = append(*elements, MarkdownElement{
			Type:     "list",
			StartPos: listStart,
			EndPos:   len(text),
			Content:  text[listStart:],
			Metadata: make(map[string]interface{}),
		})
	}
}

// parseBlockquoteElements identifies blockquote structures
func (sc *SemanticChunker) parseBlockquoteElements(text string, elements *[]MarkdownElement) {
	lines := strings.Split(text, "\n")
	var blockStart, blockEnd int
	inBlock := false

	for i, line := range lines {
		isBlockquote := sc.blockquotePattern.MatchString(line)

		if isBlockquote && !inBlock {
			// Start of blockquote
			inBlock = true
			blockStart = sc.getLineStart(text, i)
		} else if !isBlockquote && inBlock && strings.TrimSpace(line) != "" {
			// End of blockquote
			blockEnd = sc.getLineStart(text, i)
			*elements = append(*elements, MarkdownElement{
				Type:     "blockquote",
				StartPos: blockStart,
				EndPos:   blockEnd,
				Content:  text[blockStart:blockEnd],
				Metadata: make(map[string]interface{}),
			})
			inBlock = false
		}
	}

	// Handle blockquote at end of document
	if inBlock {
		*elements = append(*elements, MarkdownElement{
			Type:     "blockquote",
			StartPos: blockStart,
			EndPos:   len(text),
			Content:  text[blockStart:],
			Metadata: make(map[string]interface{}),
		})
	}
}

// findMarkdownBoundaries identifies all potential semantic boundaries
func (sc *SemanticChunker) findMarkdownBoundaries(text string, elements []MarkdownElement) []BoundaryMarker {
	var boundaries []BoundaryMarker

	// Add boundaries for Markdown elements
	for _, element := range elements {
		var score float64
		var boundaryType string

		switch element.Type {
		case "heading":
			score = sc.options.HeadingWeight
			if sc.options.HeaderHierarchy {
				// Higher level headings (H1, H2) get higher scores
				score *= (7.0 - float64(element.Level)) / 6.0
			}
			boundaryType = "heading"
		case "code_block":
			score = sc.options.CodeBlockWeight
			boundaryType = "code_block"
		case "table":
			score = sc.options.TableWeight
			boundaryType = "table"
		case "list":
			score = sc.options.ListWeight
			boundaryType = "list"
		case "blockquote":
			score = sc.options.BlockquoteWeight
			boundaryType = "blockquote"
		case "horizontal_rule":
			score = sc.options.HorizontalRuleWeight
			boundaryType = "horizontal_rule"
		default:
			continue
		}

		// Add boundary at start of element
		boundaries = append(boundaries, BoundaryMarker{
			Position:     element.StartPos,
			Type:         boundaryType,
			Score:        score,
			Context:      sc.getContext(text, element.StartPos, 50),
			HeadingLevel: element.Level,
			Metadata:     element.Metadata,
		})

		// Add boundary at end of element (with slightly lower score)
		if element.EndPos < len(text) {
			boundaries = append(boundaries, BoundaryMarker{
				Position:     element.EndPos,
				Type:         boundaryType + "_end",
				Score:        score * 0.8,
				Context:      sc.getContext(text, element.EndPos, 50),
				HeadingLevel: element.Level,
				Metadata:     element.Metadata,
			})
		}
	}

	// Add paragraph boundaries (outside of special elements)
	paragraphMatches := sc.paragraphPattern.FindAllStringIndex(text, -1)
	for _, match := range paragraphMatches {
		pos := match[1]
		if !sc.isInsideElement(pos, elements) {
			boundaries = append(boundaries, BoundaryMarker{
				Position: pos,
				Type:     "paragraph",
				Score:    sc.options.ParagraphWeight,
				Context:  sc.getContext(text, pos, 30),
				Metadata: make(map[string]interface{}),
			})
		}
	}

	// Add sentence boundaries (outside of special elements)
	sentenceMatches := sc.sentencePattern.FindAllStringIndex(text, -1)
	for _, match := range sentenceMatches {
		pos := match[1]
		if !sc.isInsideElement(pos, elements) {
			score := sc.calculateSentenceBoundaryScore(text, pos)
			boundaries = append(boundaries, BoundaryMarker{
				Position: pos,
				Type:     "sentence",
				Score:    score * sc.options.SentenceWeight,
				Context:  sc.getContext(text, pos, 50),
				Metadata: make(map[string]interface{}),
			})
		}
	}

	return boundaries
}

// generateMarkdownChunks creates final chunks respecting Markdown structure
func (sc *SemanticChunker) generateMarkdownChunks(text string, boundaries []BoundaryMarker, elements []MarkdownElement) []Chunk {
	var chunks []Chunk
	currentPos := 0

	for currentPos < len(text) {
		chunk := sc.findOptimalMarkdownChunk(text, currentPos, boundaries, elements)
		chunks = append(chunks, chunk)

		// Move to next position with overlap consideration
		nextPos := chunk.EndPos
		if sc.options.OverlapSize > 0 && nextPos < len(text) {
			overlapStart := max(chunk.StartPos, nextPos-sc.options.OverlapSize)
			// Don't create overlap inside protected elements
			if !sc.isRangeInsideProtectedElement(overlapStart, nextPos, elements) {
				nextPos = overlapStart
			}
		}

		currentPos = nextPos
		if currentPos >= chunk.EndPos {
			currentPos = chunk.EndPos
		}
	}

	return chunks
}

// findOptimalMarkdownChunk finds the best chunk respecting Markdown structure
func (sc *SemanticChunker) findOptimalMarkdownChunk(text string, startPos int, boundaries []BoundaryMarker, elements []MarkdownElement) Chunk {
	maxEnd := min(len(text), startPos+sc.options.MaxChunkSize)
	minEnd := min(len(text), startPos+sc.options.MinChunkSize)

	// Check if we're starting inside a protected element
	protectedElement := sc.getProtectedElementAt(startPos, elements)
	if protectedElement != nil {
		// Include the entire protected element
		endPos := min(maxEnd, protectedElement.EndPos)
		return Chunk{
			Text:         text[startPos:endPos],
			StartPos:     startPos,
			EndPos:       endPos,
			ChunkType:    protectedElement.Type,
			Score:        1.0,
			HeadingLevel: protectedElement.Level,
			Metadata:     protectedElement.Metadata,
		}
	}

	// Find boundaries within our range, avoiding protected elements
	var candidateBoundaries []BoundaryMarker
	for _, boundary := range boundaries {
		if boundary.Position >= minEnd && boundary.Position <= maxEnd {
			// Don't split inside protected elements
			if !sc.isInsideProtectedElement(boundary.Position, elements) {
				candidateBoundaries = append(candidateBoundaries, boundary)
			}
		}
	}

	// If no good boundaries found, use size-based chunking
	if len(candidateBoundaries) == 0 {
		endPos := maxEnd
		// Try to end at a word boundary, but respect protected elements
		for endPos > minEnd && endPos < len(text) {
			if sc.isInsideProtectedElement(endPos, elements) {
				endPos--
				continue
			}
			if unicode.IsSpace(rune(text[endPos])) {
				break
			}
			endPos--
		}

		return Chunk{
			Text:      text[startPos:endPos],
			StartPos:  startPos,
			EndPos:    endPos,
			ChunkType: "arbitrary",
			Score:     0.1,
			Metadata:  make(map[string]interface{}),
		}
	}

	// Find the best boundary based on score and position
	bestBoundary := candidateBoundaries[0]
	bestScore := sc.calculateBoundaryScore(bestBoundary, startPos)

	for _, boundary := range candidateBoundaries[1:] {
		score := sc.calculateBoundaryScore(boundary, startPos)
		if score > bestScore {
			bestBoundary = boundary
			bestScore = score
		}
	}

	return Chunk{
		Text:         text[startPos:bestBoundary.Position],
		StartPos:     startPos,
		EndPos:       bestBoundary.Position,
		ChunkType:    bestBoundary.Type,
		Score:        bestScore,
		HeadingLevel: bestBoundary.HeadingLevel,
		Metadata:     bestBoundary.Metadata,
	}
}

// calculateBoundaryScore calculates a score for a boundary considering position and type
func (sc *SemanticChunker) calculateBoundaryScore(boundary BoundaryMarker, startPos int) float64 {
	// Base score from boundary type
	score := boundary.Score

	// Position preference (prefer boundaries closer to 70% of max chunk size)
	distance := float64(boundary.Position - startPos)
	idealDistance := float64(sc.options.MaxChunkSize) * 0.7
	positionScore := 1.0 - abs(distance-idealDistance)/idealDistance

	// Boost score for heading boundaries
	if strings.HasPrefix(boundary.Type, "heading") {
		score *= 1.2
	}

	// Boost score for section-ending boundaries
	if strings.HasSuffix(boundary.Type, "_end") {
		score *= 1.1
	}

	return score * positionScore
}

// Helper functions for Markdown structure awareness

func (sc *SemanticChunker) isInsideElement(pos int, elements []MarkdownElement) bool {
	for _, element := range elements {
		if pos >= element.StartPos && pos < element.EndPos {
			return true
		}
	}
	return false
}

func (sc *SemanticChunker) isInsideProtectedElement(pos int, elements []MarkdownElement) bool {
	if !sc.options.RespectCodeBlocks && !sc.options.RespectTables {
		return false
	}

	for _, element := range elements {
		if pos >= element.StartPos && pos < element.EndPos {
			if (sc.options.RespectCodeBlocks && element.Type == "code_block") ||
				(sc.options.RespectTables && element.Type == "table") {
				return true
			}
		}
	}
	return false
}

func (sc *SemanticChunker) isRangeInsideProtectedElement(start, end int, elements []MarkdownElement) bool {
	for i := start; i < end; i++ {
		if sc.isInsideProtectedElement(i, elements) {
			return true
		}
	}
	return false
}

func (sc *SemanticChunker) getProtectedElementAt(pos int, elements []MarkdownElement) *MarkdownElement {
	if !sc.options.RespectCodeBlocks && !sc.options.RespectTables {
		return nil
	}

	for _, element := range elements {
		if pos >= element.StartPos && pos < element.EndPos {
			if (sc.options.RespectCodeBlocks && element.Type == "code_block") ||
				(sc.options.RespectTables && element.Type == "table") {
				return &element
			}
		}
	}
	return nil
}

func (sc *SemanticChunker) getLineStart(text string, lineNum int) int {
	lines := strings.Split(text, "\n")
	pos := 0
	for i := 0; i < lineNum && i < len(lines); i++ {
		pos += len(lines[i]) + 1 // +1 for newline
	}
	return pos
}

func (sc *SemanticChunker) getLineNumber(text string, pos int) int {
	return strings.Count(text[:pos], "\n")
}

// Reuse helper functions from original implementation
func (sc *SemanticChunker) getContext(text string, pos int, contextSize int) string {
	start := max(0, pos-contextSize/2)
	end := min(len(text), pos+contextSize/2)
	return text[start:end]
}

func (sc *SemanticChunker) calculateSentenceBoundaryScore(text string, pos int) float64 {
	baseScore := 1.0

	// Get surrounding context
	start := max(0, pos-100)
	end := min(len(text), pos+100)
	context := text[start:end]

	// Lower score for abbreviations
	abbrevPattern := regexp.MustCompile(`\b[A-Z][a-z]*\.\s*$`)
	if abbrevPattern.MatchString(text[max(0, pos-20):pos]) {
		baseScore *= 0.3
	}

	// Lower score for numbers with decimals
	numberPattern := regexp.MustCompile(`\d+\.\d+`)
	if numberPattern.MatchString(text[max(0, pos-10):min(len(text), pos+10)]) {
		baseScore *= 0.2
	}

	// Higher score if followed by capitalized word (but not inside code)
	if pos < len(text)-1 && !strings.Contains(context, "`") {
		nextWord := sc.getNextWord(text, pos)
		if len(nextWord) > 0 && unicode.IsUpper(rune(nextWord[0])) {
			baseScore *= 1.2
		}
	}

	return baseScore
}

func (sc *SemanticChunker) getNextWord(text string, pos int) string {
	// Skip whitespace
	for pos < len(text) && unicode.IsSpace(rune(text[pos])) {
		pos++
	}

	// Extract word
	start := pos
	for pos < len(text) && !unicode.IsSpace(rune(text[pos])) {
		pos++
	}

	if start >= len(text) {
		return ""
	}

	return text[start:pos]
}

// Utility functions
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func abs(a float64) float64 {
	if a < 0 {
		return -a
	}
	return a
}

// Vibe implements DocumentHandler interface for semantic chunking with Markdown awareness
type Vibe struct {
	ChunkingOptions ChunkingOptions
	
	// Entity extraction configuration
	EntityExtractionGoal     string
	EntityTypes              []string
	Language                 string
	EntityExtractionExamples []golightrag.EntityExtractionPromptExample

	// Configuration for RAG operations
	Config DocumentConfig
}

// NewVibe creates a new Vibe handler with default Markdown chunking options
func NewVibe() *Vibe {
	return &Vibe{
		ChunkingOptions: DefaultMarkdownChunkingOptions(),
		Language:        defaultLanguage,
		Config: DocumentConfig{
			BackoffDuration:  defaultBackoffDuration,
			ConcurrencyCount: defaultConcurrencyCount,
		},
	}
}

// ChunksDocument implements DocumentHandler.ChunksDocument using semantic chunking
func (v *Vibe) ChunksDocument(content string) ([]golightrag.Source, error) {
	if content == "" {
		return []golightrag.Source{}, nil
	}

	// Create semantic chunker with configured options
	chunker := NewMarkdownChunker(v.ChunkingOptions)
	
	// Perform semantic chunking
	semanticChunks := chunker.ChunkMarkdown(content)
	
	// Convert to golightrag.Source format
	results := make([]golightrag.Source, len(semanticChunks))
	for i, chunk := range semanticChunks {
		// Count tokens for the chunk content
		tokenCount, err := internal.CountTokens(chunk.Text)
		if err != nil {
			return nil, fmt.Errorf("failed to count tokens for chunk %d: %w", i, err)
		}
		
		results[i] = golightrag.Source{
			Content:    strings.TrimSpace(chunk.Text),
			TokenSize:  tokenCount,
			OrderIndex: chunk.StartPos, // Use start position as order index for semantic chunks
		}
	}
	
	return results, nil
}

// EntityExtractionPromptData implements DocumentHandler.EntityExtractionPromptData
func (v *Vibe) EntityExtractionPromptData() golightrag.EntityExtractionPromptData {
	goal := v.EntityExtractionGoal
	if goal == "" {
		goal = defaultEntityExtractionGoal
	}
	entityTypes := v.EntityTypes
	if entityTypes == nil {
		entityTypes = defaultEntityTypes
	}
	language := v.Language
	if language == "" {
		language = defaultLanguage
	}
	examples := v.EntityExtractionExamples
	if examples == nil {
		examples = defaultEntityExtractionExamples
	}
	return golightrag.EntityExtractionPromptData{
		Goal:        goal,
		EntityTypes: entityTypes,
		Language:    language,
		Examples:    examples,
	}
}

// MaxRetries implements DocumentHandler.MaxRetries
func (v *Vibe) MaxRetries() int {
	return v.Config.MaxRetries
}

// BackoffDuration implements DocumentHandler.BackoffDuration
func (v *Vibe) BackoffDuration() time.Duration {
	if v.Config.BackoffDuration == 0 {
		return defaultBackoffDuration
	}
	return v.Config.BackoffDuration
}

// ConcurrencyCount implements DocumentHandler.ConcurrencyCount
func (v *Vibe) ConcurrencyCount() int {
	if v.Config.ConcurrencyCount == 0 {
		return defaultConcurrencyCount
	}
	return v.Config.ConcurrencyCount
}

// GleanCount implements DocumentHandler.GleanCount
func (v *Vibe) GleanCount() int {
	return v.Config.GleanCount
}

// MaxSummariesTokenLength implements DocumentHandler.MaxSummariesTokenLength
func (v *Vibe) MaxSummariesTokenLength() int {
	if v.Config.MaxSummariesTokenLength == 0 {
		return defaultMaxSummariesTokenLength
	}
	return v.Config.MaxSummariesTokenLength
}

// DisplayChunkInfo provides detailed information about chunks for debugging
func DisplayChunkInfo(chunks []Chunk) {
	fmt.Printf("Generated %d chunks:\n\n", len(chunks))

	for i, chunk := range chunks {
		fmt.Printf("=== Chunk %d ===\n", i+1)
		fmt.Printf("Type: %s\n", chunk.ChunkType)
		fmt.Printf("Score: %.3f\n", chunk.Score)
		fmt.Printf("Position: %d-%d (%d chars)\n", chunk.StartPos, chunk.EndPos, len(chunk.Text))

		if chunk.HeadingLevel > 0 {
			fmt.Printf("Heading Level: H%d\n", chunk.HeadingLevel)
		}

		if len(chunk.Metadata) > 0 {
			fmt.Printf("Metadata: %+v\n", chunk.Metadata)
		}

		// Show first few lines of content
		lines := strings.Split(strings.TrimSpace(chunk.Text), "\n")
		maxLines := min(5, len(lines))
		fmt.Printf("Content preview:\n")
		for j := 0; j < maxLines; j++ {
			fmt.Printf("  %s\n", lines[j])
		}
		if len(lines) > maxLines {
			fmt.Printf("  ... (%d more lines)\n", len(lines)-maxLines)
		}
		fmt.Printf("\n")
	}
}
