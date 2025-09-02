package handler

import (
	"fmt"
	"regexp"
	"strings"
	"time"
	"unicode"

	golightrag "github.com/MegaGrindStone/go-light-rag"
	"github.com/MegaGrindStone/go-light-rag/internal"
	"github.com/yuin/goldmark"
	"github.com/yuin/goldmark/ast"
	"github.com/yuin/goldmark/text"
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

// Section represents a document section defined by headings
type Section struct {
	Heading  *MarkdownElement  // The heading that starts this section (nil for implicit section)
	Content  []MarkdownElement // All content elements within this section
	StartPos int
	EndPos   int
	Level    int    // Heading level (0 for implicit sections)
	Text     string // Raw text of the entire section
}

// ASTChunker handles AST-based chunking with section awareness
type ASTChunker struct {
	options ChunkingOptions
	parser  goldmark.Markdown
}

// NewASTChunker creates a new AST-based chunker optimized for Markdown sections
func NewASTChunker(options ChunkingOptions) *ASTChunker {
	return &ASTChunker{
		options: options,
		parser:  goldmark.New(),
	}
}

// NewMarkdownChunker creates a new chunker optimized for Markdown (backward compatibility)
func NewMarkdownChunker(options ChunkingOptions) *ASTChunker {
	return NewASTChunker(options)
}

// ChunkMarkdown performs AST-based section-aware chunking on Markdown text
func (ac *ASTChunker) ChunkMarkdown(content string) []Chunk {
	if len(content) <= ac.options.MaxChunkSize {
		return []Chunk{{
			Text:      content,
			StartPos:  0,
			EndPos:    len(content),
			ChunkType: "complete",
			Score:     1.0,
			Metadata:  make(map[string]interface{}),
		}}
	}

	// Parse the markdown into an AST
	source := []byte(content)
	reader := text.NewReader(source)
	doc := ac.parser.Parser().Parse(reader)

	// Extract sections from AST
	sections := ac.extractSections(doc, source)

	// Generate chunks by processing sections and splitting on paragraphs
	return ac.chunkBySections(sections, content)
}

// extractSections parses the AST and extracts document sections
func (ac *ASTChunker) extractSections(doc ast.Node, source []byte) []Section {
	var sections []Section
	var currentSection *Section

	// Walk the AST to build sections
	err := ast.Walk(doc, func(node ast.Node, entering bool) (ast.WalkStatus, error) {
		if !entering {
			return ast.WalkContinue, nil
		}

		switch n := node.(type) {
		case *ast.Heading:
			// Get the position from the node's lines
			lines := n.Lines()
			if lines.Len() == 0 {
				return ast.WalkContinue, nil
			}
			nodeStart := lines.At(0).Start

			// Save the current section if it exists
			if currentSection != nil {
				currentSection.EndPos = nodeStart
				currentSection.Text = string(source[currentSection.StartPos:currentSection.EndPos])
				sections = append(sections, *currentSection)
			}

			// Create a new section for this heading
			headingElement := ac.astNodeToElement(n, source)
			currentSection = &Section{
				Heading:  &headingElement,
				Content:  []MarkdownElement{headingElement},
				StartPos: nodeStart,
				Level:    n.Level,
			}

		default:
			// Skip inline nodes entirely - check for common inline node types
			switch node.(type) {
			case *ast.Text, *ast.CodeSpan, *ast.Emphasis, *ast.Link, *ast.Image:
				return ast.WalkContinue, nil
			}

			// Add this element to the current section
			if currentSection == nil {
				// Create an implicit section for content before the first heading
				currentSection = &Section{
					Heading:  nil,
					Content:  []MarkdownElement{},
					StartPos: 0,
					Level:    0,
				}
			}

			element := ac.astNodeToElement(node, source)
			if element.Type != "unknown" {
				currentSection.Content = append(currentSection.Content, element)
			}
		}

		return ast.WalkContinue, nil
	})

	if err != nil {
		// Fallback to treating entire document as one section
		return []Section{{
			Heading:  nil,
			Content:  []MarkdownElement{},
			StartPos: 0,
			EndPos:   len(source),
			Level:    0,
			Text:     string(source),
		}}
	}

	// Add the final section
	if currentSection != nil {
		currentSection.EndPos = len(source)
		currentSection.Text = string(source[currentSection.StartPos:currentSection.EndPos])
		sections = append(sections, *currentSection)
	}

	return sections
}

// astNodeToElement converts an AST node to a MarkdownElement
func (ac *ASTChunker) astNodeToElement(node ast.Node, source []byte) MarkdownElement {
	// Handle different types of nodes (block vs inline)
	var start, stop int

	// Check if this is a block node (has Lines method) or inline node
	if blockNode, ok := node.(interface{ Lines() *text.Segments }); ok {
		lines := blockNode.Lines()
		if lines.Len() == 0 {
			return MarkdownElement{
				Type:     "unknown",
				StartPos: 0,
				EndPos:   0,
				Content:  "",
				Metadata: make(map[string]interface{}),
			}
		}

		segment := lines.At(0)
		start = segment.Start
		stop = segment.Stop

		// For multi-line elements, get the full range
		for i := 1; i < lines.Len(); i++ {
			seg := lines.At(i)
			if seg.Stop > stop {
				stop = seg.Stop
			}
		}
	} else {
		// For inline nodes, skip them in section extraction
		return MarkdownElement{
			Type:     "inline",
			StartPos: 0,
			EndPos:   0,
			Content:  "",
			Metadata: make(map[string]interface{}),
		}
	}

	content := string(source[start:stop])

	switch n := node.(type) {
	case *ast.Heading:
		return MarkdownElement{
			Type:     "heading",
			StartPos: start,
			EndPos:   stop,
			Level:    n.Level,
			Content:  content,
			Metadata: map[string]interface{}{"level": n.Level},
		}

	case *ast.CodeBlock, *ast.FencedCodeBlock:
		return MarkdownElement{
			Type:     "code_block",
			StartPos: start,
			EndPos:   stop,
			Content:  content,
			Metadata: make(map[string]interface{}),
		}

	case *ast.List:
		return MarkdownElement{
			Type:     "list",
			StartPos: start,
			EndPos:   stop,
			Content:  content,
			Metadata: make(map[string]interface{}),
		}

	case *ast.Blockquote:
		return MarkdownElement{
			Type:     "blockquote",
			StartPos: start,
			EndPos:   stop,
			Content:  content,
			Metadata: make(map[string]interface{}),
		}

	case *ast.Paragraph:
		return MarkdownElement{
			Type:     "paragraph",
			StartPos: start,
			EndPos:   stop,
			Content:  content,
			Metadata: make(map[string]interface{}),
		}

	case *ast.ThematicBreak:
		return MarkdownElement{
			Type:     "horizontal_rule",
			StartPos: start,
			EndPos:   stop,
			Content:  content,
			Metadata: make(map[string]interface{}),
		}

	default:
		return MarkdownElement{
			Type:     "unknown",
			StartPos: start,
			EndPos:   stop,
			Content:  content,
			Metadata: make(map[string]interface{}),
		}
	}
}

// chunkBySections processes sections and splits them by paragraphs while preserving structure
func (ac *ASTChunker) chunkBySections(sections []Section, fullText string) []Chunk {
	var chunks []Chunk

	for _, section := range sections {
		if len(section.Text) <= ac.options.MaxChunkSize {
			// Section fits in one chunk
			chunks = append(chunks, Chunk{
				Text:         strings.TrimSpace(section.Text),
				StartPos:     section.StartPos,
				EndPos:       section.EndPos,
				ChunkType:    "section",
				Score:        1.0,
				HeadingLevel: section.Level,
				Metadata:     map[string]interface{}{"section": true},
			})
		} else {
			// Section too large, split by paragraphs within section
			sectionChunks := ac.chunkSectionByParagraphs(section)
			chunks = append(chunks, sectionChunks...)
		}
	}

	return chunks
}

// chunkSectionByParagraphs splits a section into chunks at paragraph boundaries
func (ac *ASTChunker) chunkSectionByParagraphs(section Section) []Chunk {
	var chunks []Chunk
	text := section.Text

	// Find paragraph boundaries within the section
	paragraphs := ac.findParagraphBoundaries(text)

	if len(paragraphs) <= 1 {
		// No paragraph boundaries found, or only one paragraph
		// Split by sentences as fallback while preserving word boundaries
		return ac.chunkSectionBySentences(section)
	}

	currentStart := 0
	currentContent := ""

	for _, paragraphEnd := range paragraphs {
		paragraphText := text[currentStart:paragraphEnd]

		// Check if adding this paragraph would exceed chunk size
		if len(currentContent) > 0 && len(currentContent)+len(paragraphText) > ac.options.MaxChunkSize {
			// Finalize current chunk
			if len(currentContent) >= ac.options.MinChunkSize || len(chunks) == 0 {
				chunks = append(chunks, Chunk{
					Text:         strings.TrimSpace(currentContent),
					StartPos:     section.StartPos + (currentStart - len(currentContent)),
					EndPos:       section.StartPos + currentStart,
					ChunkType:    "section_paragraph",
					Score:        ac.options.ParagraphWeight,
					HeadingLevel: section.Level,
					Metadata:     map[string]interface{}{"section": true, "split_method": "paragraph"},
				})
			}
			// Start new chunk
			currentContent = paragraphText
		} else {
			// Add paragraph to current chunk
			currentContent += paragraphText
		}

		currentStart = paragraphEnd
	}

	// Add final chunk if there's remaining content
	if len(strings.TrimSpace(currentContent)) > 0 {
		chunks = append(chunks, Chunk{
			Text:         strings.TrimSpace(currentContent),
			StartPos:     section.StartPos + (currentStart - len(currentContent)),
			EndPos:       section.EndPos,
			ChunkType:    "section_paragraph",
			Score:        ac.options.ParagraphWeight,
			HeadingLevel: section.Level,
			Metadata:     map[string]interface{}{"section": true, "split_method": "paragraph"},
		})
	}

	return chunks
}

// findParagraphBoundaries identifies paragraph boundaries within text
func (ac *ASTChunker) findParagraphBoundaries(text string) []int {
	var boundaries []int

	// Find double newlines that indicate paragraph breaks
	paragraphPattern := regexp.MustCompile(`\n\s*\n`)
	matches := paragraphPattern.FindAllStringIndex(text, -1)

	for _, match := range matches {
		// Use the end of the paragraph break as the boundary
		boundaries = append(boundaries, match[1])
	}

	// Always include the end of the text as a boundary
	if len(boundaries) == 0 || boundaries[len(boundaries)-1] != len(text) {
		boundaries = append(boundaries, len(text))
	}

	return boundaries
}

// chunkSectionBySentences is a fallback when no paragraph boundaries are found
func (ac *ASTChunker) chunkSectionBySentences(section Section) []Chunk {
	var chunks []Chunk
	text := section.Text

	// Find sentence boundaries
	sentencePattern := regexp.MustCompile(`[.!?]+(?:\s+|$)`)
	sentenceBoundaries := ac.findSentenceBoundaries(text, sentencePattern)

	if len(sentenceBoundaries) <= 1 {
		// No sentence boundaries, split on word boundaries as last resort
		return ac.chunkSectionByWords(section)
	}

	currentStart := 0
	currentContent := ""

	for _, sentenceEnd := range sentenceBoundaries {
		sentenceText := text[currentStart:sentenceEnd]

		// Check if adding this sentence would exceed chunk size
		if len(currentContent) > 0 && len(currentContent)+len(sentenceText) > ac.options.MaxChunkSize {
			// Finalize current chunk
			if len(currentContent) >= ac.options.MinChunkSize || len(chunks) == 0 {
				chunks = append(chunks, Chunk{
					Text:         strings.TrimSpace(currentContent),
					StartPos:     section.StartPos + (currentStart - len(currentContent)),
					EndPos:       section.StartPos + currentStart,
					ChunkType:    "section_sentence",
					Score:        ac.options.SentenceWeight,
					HeadingLevel: section.Level,
					Metadata:     map[string]interface{}{"section": true, "split_method": "sentence"},
				})
			}
			// Start new chunk
			currentContent = sentenceText
		} else {
			// Add sentence to current chunk
			currentContent += sentenceText
		}

		currentStart = sentenceEnd
	}

	// Add final chunk if there's remaining content
	if len(strings.TrimSpace(currentContent)) > 0 {
		chunks = append(chunks, Chunk{
			Text:         strings.TrimSpace(currentContent),
			StartPos:     section.StartPos + (currentStart - len(currentContent)),
			EndPos:       section.EndPos,
			ChunkType:    "section_sentence",
			Score:        ac.options.SentenceWeight,
			HeadingLevel: section.Level,
			Metadata:     map[string]interface{}{"section": true, "split_method": "sentence"},
		})
	}

	return chunks
}

// findSentenceBoundaries finds sentence boundaries while avoiding abbreviations
func (ac *ASTChunker) findSentenceBoundaries(text string, pattern *regexp.Regexp) []int {
	var boundaries []int
	matches := pattern.FindAllStringIndex(text, -1)

	for _, match := range matches {
		pos := match[1]
		// Skip boundaries that look like abbreviations or decimals
		if ac.isValidSentenceBoundary(text, pos) {
			boundaries = append(boundaries, pos)
		}
	}

	// Always include the end of the text as a boundary
	if len(boundaries) == 0 || boundaries[len(boundaries)-1] != len(text) {
		boundaries = append(boundaries, len(text))
	}

	return boundaries
}

// isValidSentenceBoundary checks if a potential sentence boundary is valid
func (ac *ASTChunker) isValidSentenceBoundary(text string, pos int) bool {
	// Lower score for abbreviations
	abbrevPattern := regexp.MustCompile(`\b[A-Z][a-z]*\.\s*$`)
	if abbrevPattern.MatchString(text[max(0, pos-20):pos]) {
		return false
	}

	// Lower score for numbers with decimals
	numberPattern := regexp.MustCompile(`\d+\.\d+`)
	if numberPattern.MatchString(text[max(0, pos-10):min(len(text), pos+10)]) {
		return false
	}

	return true
}

// chunkSectionByWords is the final fallback for sections with no sentence boundaries
func (ac *ASTChunker) chunkSectionByWords(section Section) []Chunk {
	text := section.Text

	// Simple word-boundary chunking as absolute fallback
	chunkSize := ac.options.MaxChunkSize
	if len(text) <= chunkSize {
		return []Chunk{{
			Text:         strings.TrimSpace(text),
			StartPos:     section.StartPos,
			EndPos:       section.EndPos,
			ChunkType:    "section_word",
			Score:        0.1,
			HeadingLevel: section.Level,
			Metadata:     map[string]interface{}{"section": true, "split_method": "word"},
		}}
	}

	var chunks []Chunk
	for i := 0; i < len(text); i += chunkSize {
		end := min(i+chunkSize, len(text))

		// Try to end on word boundary
		if end < len(text) {
			for end > i+ac.options.MinChunkSize && end < len(text) && !unicode.IsSpace(rune(text[end])) {
				end--
			}
		}

		chunkText := text[i:end]
		chunks = append(chunks, Chunk{
			Text:         strings.TrimSpace(chunkText),
			StartPos:     section.StartPos + i,
			EndPos:       section.StartPos + end,
			ChunkType:    "section_word",
			Score:        0.1,
			HeadingLevel: section.Level,
			Metadata:     map[string]interface{}{"section": true, "split_method": "word"},
		})
	}

	return chunks
}

// hasActualContent checks if a chunk contains meaningful content beyond markdown syntax
func hasActualContent(content string) bool {
	if content == "" {
		return false
	}

	// Remove common markdown syntax patterns
	cleaned := content

	// Remove heading markers (# ## ### etc.)
	headingPattern := regexp.MustCompile(`^#{1,6}\s*$`)
	if headingPattern.MatchString(strings.TrimSpace(cleaned)) {
		return false
	}

	// Remove horizontal rules (--- === ***)
	hrPattern := regexp.MustCompile(`^[-=*]{3,}\s*$`)
	if hrPattern.MatchString(strings.TrimSpace(cleaned)) {
		return false
	}

	// Remove list markers and check if anything remains
	listPattern := regexp.MustCompile(`^[\s]*[-*+]\s*$|^[\s]*\d+\.\s*$`)
	if listPattern.MatchString(strings.TrimSpace(cleaned)) {
		return false
	}

	// Remove blockquote markers
	blockquotePattern := regexp.MustCompile(`^>\s*$`)
	if blockquotePattern.MatchString(strings.TrimSpace(cleaned)) {
		return false
	}

	// Remove code block markers
	codeBlockPattern := regexp.MustCompile("^```\\s*$|^~~~\\s*$")
	if codeBlockPattern.MatchString(strings.TrimSpace(cleaned)) {
		return false
	}

	// Check if content contains actual text after removing markdown syntax
	// Remove all markdown syntax and see if substantial text remains
	syntaxPattern := regexp.MustCompile(`[#\-=*+>~` + "`" + `\[\](){}|\\_]`)
	cleanedText := syntaxPattern.ReplaceAllString(cleaned, "")
	cleanedText = regexp.MustCompile(`\s+`).ReplaceAllString(cleanedText, " ")
	cleanedText = strings.TrimSpace(cleanedText)

	// Require at least some meaningful text (more than just single characters or numbers)
	if len(cleanedText) < 3 {
		return false
	}

	// Check if it's just whitespace, numbers, or single characters
	if regexp.MustCompile(`^[\s\d.,;:!?\-]*$`).MatchString(cleanedText) {
		return false
	}

	return true
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

// MarkdownAst implements DocumentHandler interface for AST-based chunking with Markdown section awareness
type MarkdownAst struct {
	ChunkingOptions ChunkingOptions

	// Entity extraction configuration
	EntityExtractionGoal     string
	EntityTypes              []string
	Language                 string
	EntityExtractionExamples []golightrag.EntityExtractionPromptExample

	// Configuration for RAG operations
	Config DocumentConfig
}

// NewMarkdownAst creates a new MarkdownAst handler with default Markdown chunking options
func NewMarkdownAst(config DocumentConfig) *MarkdownAst {
	return &MarkdownAst{
		ChunkingOptions: DefaultMarkdownChunkingOptions(),
		Language:        defaultLanguage,
		Config:          config,
	}
}

// ChunksDocument implements DocumentHandler.ChunksDocument using AST-based section-aware chunking
func (m *MarkdownAst) ChunksDocument(content string) ([]golightrag.Source, error) {
	if content == "" {
		return []golightrag.Source{}, nil
	}

	// Create AST-based chunker with configured options
	chunker := NewASTChunker(m.ChunkingOptions)

	// Perform section-aware chunking
	sectionChunks := chunker.ChunkMarkdown(content)

	// Convert to golightrag.Source format, filtering out empty or syntax-only chunks
	var results []golightrag.Source
	for _, chunk := range sectionChunks {
		// Trim the content first
		trimmedContent := strings.TrimSpace(chunk.Text)

		// Skip chunks that don't have actual content
		if !hasActualContent(trimmedContent) {
			continue
		}

		tokenCount, err := internal.CountTokens(trimmedContent)
		if err != nil {
			return nil, fmt.Errorf("failed to count tokens for chunk: %w", err)
		}

		results = append(results, golightrag.Source{
			Content:    trimmedContent,
			TokenSize:  tokenCount,
			OrderIndex: chunk.StartPos, // Use start position as order index for section chunks
		})
	}

	return results, nil
}

// EntityExtractionPromptData implements DocumentHandler.EntityExtractionPromptData
func (m *MarkdownAst) EntityExtractionPromptData() golightrag.EntityExtractionPromptData {
	goal := m.EntityExtractionGoal
	if goal == "" {
		goal = defaultEntityExtractionGoal
	}
	entityTypes := m.EntityTypes
	if entityTypes == nil {
		entityTypes = defaultEntityTypes
	}
	language := m.Language
	if language == "" {
		language = defaultLanguage
	}
	examples := m.EntityExtractionExamples
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
func (m *MarkdownAst) MaxRetries() int {
	return m.Config.MaxRetries
}

// BackoffDuration implements DocumentHandler.BackoffDuration
func (m *MarkdownAst) BackoffDuration() time.Duration {
	if m.Config.BackoffDuration == 0 {
		return defaultBackoffDuration
	}
	return m.Config.BackoffDuration
}

// ConcurrencyCount implements DocumentHandler.ConcurrencyCount
func (m *MarkdownAst) ConcurrencyCount() int {
	if m.Config.ConcurrencyCount == 0 {
		return defaultConcurrencyCount
	}
	return m.Config.ConcurrencyCount
}

// GleanCount implements DocumentHandler.GleanCount
func (m *MarkdownAst) GleanCount() int {
	return m.Config.GleanCount
}

// MaxSummariesTokenLength implements DocumentHandler.MaxSummariesTokenLength
func (m *MarkdownAst) MaxSummariesTokenLength() int {
	if m.Config.MaxSummariesTokenLength == 0 {
		return defaultMaxSummariesTokenLength
	}
	return m.Config.MaxSummariesTokenLength
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
