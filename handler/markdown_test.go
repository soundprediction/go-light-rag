package handler_test

import (
	"strings"
	"testing"
	"unicode"

	golightrag "github.com/MegaGrindStone/go-light-rag"
	"github.com/MegaGrindStone/go-light-rag/handler"
	"github.com/MegaGrindStone/go-light-rag/internal"
)

func TestMarkdownAst_ChunksDocument(t *testing.T) {
	tests := []struct {
		name             string
		content          string
		handlerConfig    *handler.MarkdownAst
		expectedChunks   int
		expectError      bool
		verificationFunc func(t *testing.T, chunks []golightrag.Source)
	}{
		{
			name:           "Empty content",
			content:        "",
			expectedChunks: 0,
		},
		{
			name:           "Small content within single chunk",
			content:        "This is a small text that should fit in a single chunk.",
			expectedChunks: 1,
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				verifyChunkCount(t, chunks, 1)
				if chunks[0].Content != "This is a small text that should fit in a single chunk." {
					t.Errorf("Content mismatch: %s", chunks[0].Content)
				}
			},
		},
		{
			name: "Markdown with headings",
			content: `# Main Title

This is the introduction paragraph.

## Section 1

Content for section 1 with multiple sentences. This should be chunked appropriately based on the semantic boundaries.

## Section 2

Content for section 2. This section also has multiple sentences to test the chunking logic.`,
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) == 0 {
					t.Fatal("Expected at least one chunk")
				}
				
				// Verify chunks contain heading information
				foundMainTitle := false
				for _, chunk := range chunks {
					if strings.Contains(chunk.Content, "# Main Title") {
						foundMainTitle = true
						break
					}
				}
				if !foundMainTitle {
					t.Error("Expected to find main title in chunks")
				}
			},
		},
		{
			name: "Code blocks are preserved",
			content: `# Code Example

Here is some code:

` + "```go" + `
func main() {
    fmt.Println("Hello, World!")
    // This code block should be preserved as one unit
    for i := 0; i < 10; i++ {
        fmt.Println(i)
    }
}
` + "```" + `

And some text after the code block.`,
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) == 0 {
					t.Fatal("Expected at least one chunk")
				}
				
				// Find chunk containing code block
				foundCodeBlock := false
				for _, chunk := range chunks {
					if strings.Contains(chunk.Content, "func main()") {
						foundCodeBlock = true
						// Code block should be complete in the chunk
						if !strings.Contains(chunk.Content, "fmt.Println(i)") {
							t.Error("Code block appears to be split inappropriately")
						}
						break
					}
				}
				if !foundCodeBlock {
					t.Error("Expected to find code block in chunks")
				}
			},
		},
		{
			name: "Tables are preserved",
			content: `# Data Table

| Name | Age | City |
|------|-----|------|
| John | 30  | NYC  |
| Jane | 25  | LA   |
| Bob  | 35  | CHI  |

Some text after the table.`,
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) == 0 {
					t.Fatal("Expected at least one chunk")
				}
				
				// Find chunk containing table
				foundTable := false
				for _, chunk := range chunks {
					if strings.Contains(chunk.Content, "| Name | Age | City |") {
						foundTable = true
						// Table should be complete
						if !strings.Contains(chunk.Content, "| Bob  | 35  | CHI  |") {
							t.Error("Table appears to be split inappropriately")
						}
						break
					}
				}
				if !foundTable {
					t.Error("Expected to find table in chunks")
				}
			},
		},
		{
			name: "Lists are handled appropriately",
			content: `# Todo List

Here are my tasks:

- Task 1: Complete the project
- Task 2: Review documentation
  - Subtask 2.1: Check formatting
  - Subtask 2.2: Verify examples
- Task 3: Submit for review

End of list.`,
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) == 0 {
					t.Fatal("Expected at least one chunk")
				}
				
				// Check that list structure is preserved
				foundList := false
				for _, chunk := range chunks {
					if strings.Contains(chunk.Content, "- Task 1:") {
						foundList = true
						break
					}
				}
				if !foundList {
					t.Error("Expected to find list in chunks")
				}
			},
		},
		{
			name: "Large content with custom chunk size",
			content: strings.Repeat("This sentence contains about nine tokens. ", 200), // ~1800 tokens
			handlerConfig: &handler.MarkdownAst{
				ChunkingOptions: handler.ChunkingOptions{
					MaxChunkSize: 500,  // Smaller chunks
					MinChunkSize: 100,
					OverlapSize:  20,
				},
			},
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) < 2 {
					t.Fatalf("Expected multiple chunks with small chunk size, got %d", len(chunks))
				}
				
				// Check that chunks don't exceed max size (in characters, not tokens)
				for i, chunk := range chunks {
					if len(chunk.Content) > 500 {
						t.Errorf("Chunk %d exceeds max character size: %d > 500", i, len(chunk.Content))
					}
				}
			},
		},
		{
			name: "Unicode and special characters",
			content: "Special characters: 🚀 😊 üñîçødé\nNew lines\tTabs中文日本語\n\n# Header with émojis 🎉",
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				verifyChunkCount(t, chunks, 1)
				
				// Check all special characters are preserved
				if !strings.Contains(chunks[0].Content, "🚀") ||
					!strings.Contains(chunks[0].Content, "üñîçødé") ||
					!strings.Contains(chunks[0].Content, "中文") ||
					!strings.Contains(chunks[0].Content, "🎉") {
					t.Errorf("Special characters not preserved: %s", chunks[0].Content)
				}
			},
		},
		{
			name: "Sentence boundaries are preserved",
			content: strings.Repeat("This is the first sentence in this test case. ", 10) + 
				    strings.Repeat("This is the second sentence that should not be split. ", 10) +
				    strings.Repeat("This is the third sentence with proper punctuation! ", 10) +
				    strings.Repeat("Finally, this is the last sentence in this long text? ", 10),
			handlerConfig: &handler.MarkdownAst{
				ChunkingOptions: handler.ChunkingOptions{
					MaxChunkSize: 200,  // Force chunking
					MinChunkSize: 50,
					OverlapSize:  10,
					SentenceWeight: 0.8, // Give high priority to sentence boundaries
				},
			},
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				if len(chunks) < 2 {
					t.Fatalf("Expected multiple chunks to test sentence boundaries, got %d", len(chunks))
				}
				
				// Verify no chunks end with incomplete sentences
				for i, chunk := range chunks {
					content := strings.TrimSpace(chunk.Content)
					if len(content) == 0 {
						continue
					}
					
					// Check that chunk ends with proper sentence ending
					lastChar := content[len(content)-1]
					if lastChar != '.' && lastChar != '!' && lastChar != '?' {
						// Allow for the last chunk to not end with punctuation if it's the end of the document
						if i != len(chunks)-1 {
							t.Errorf("Chunk %d does not end with sentence punctuation: '%c' (chunk: %q)", 
								i, lastChar, content[max(0, len(content)-50):])
						}
					}
				}
			},
		},
		{
			name: "Abbreviations and decimals don't break sentence detection", 
			content: "Dr. Smith lives at 123 Main St. He has a 3.14159 acre property. Mrs. Johnson lives next door. The property is worth $1.5 million dollars.",
			handlerConfig: &handler.MarkdownAst{
				ChunkingOptions: handler.ChunkingOptions{
					MaxChunkSize: 80,  // Force chunking to test sentence boundaries
					MinChunkSize: 20,
					SentenceWeight: 0.9,
				},
			},
			verificationFunc: func(t *testing.T, chunks []golightrag.Source) {
				// Join all chunks back together and verify the content is preserved
				var reconstructed strings.Builder
				for _, chunk := range chunks {
					reconstructed.WriteString(chunk.Content)
				}
				
				// Check that key phrases are not split
				fullText := reconstructed.String()
				if !strings.Contains(fullText, "Dr. Smith") ||
					!strings.Contains(fullText, "3.14159 acre") ||
					!strings.Contains(fullText, "Mrs. Johnson") ||
					!strings.Contains(fullText, "$1.5 million") {
					t.Error("Important phrases were split across chunks incorrectly")
				}
				
				// Verify sentences aren't split inappropriately
				for i, chunk := range chunks {
					content := strings.TrimSpace(chunk.Content)
					if len(content) == 0 {
						continue
					}
					
					// Should not start mid-sentence (except first chunk)
					if i > 0 && len(content) > 0 && unicode.IsLower(rune(content[0])) {
						// Allow continuation if previous chunk ended without punctuation
						prevContent := strings.TrimSpace(chunks[i-1].Content)
						if len(prevContent) > 0 {
							lastChar := prevContent[len(prevContent)-1]
							if lastChar == '.' || lastChar == '!' || lastChar == '?' {
								t.Errorf("Chunk %d appears to start mid-sentence: %q", i, content[:min(20, len(content))])
							}
						}
					}
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Use provided handler config or default
			var h *handler.MarkdownAst
			if tt.handlerConfig != nil {
				h = tt.handlerConfig
			} else {
				h = handler.NewMarkdownAst(handler.DocumentConfig{})
			}

			chunks, err := h.ChunksDocument(tt.content)

			// Check error expectations
			if (err != nil) != tt.expectError {
				t.Errorf("ChunksDocument() error = %v, expectError = %v", err, tt.expectError)
				return
			}

			if tt.expectError {
				return
			}

			// Check number of chunks if expected
			if tt.expectedChunks > 0 && len(chunks) != tt.expectedChunks {
				t.Errorf("Expected %d chunks, got %d", tt.expectedChunks, len(chunks))
			}

			// Verify token sizes make sense
			for i, chunk := range chunks {
				// Token size should be positive for non-empty chunks
				if len(chunk.Content) > 0 && chunk.TokenSize <= 0 {
					t.Errorf("Chunk %d has invalid token size: %d", i, chunk.TokenSize)
				}

				// Verify token counts match content
				expectedTokens, err := internal.CountTokens(chunk.Content)
				if err != nil {
					t.Errorf("Failed to count tokens for verification: %v", err)
				}
				if chunk.TokenSize != expectedTokens {
					t.Errorf("Chunk %d: reported TokenSize %d doesn't match actual count %d",
						i, chunk.TokenSize, expectedTokens)
				}
			}

			// Run custom verification if provided
			if tt.verificationFunc != nil {
				tt.verificationFunc(t, chunks)
			}
		})
	}
}

func TestMarkdownAst_EntityExtractionPromptData(t *testing.T) {
	tests := []struct {
		name        string
		markdownAst *handler.MarkdownAst
		expected func(data golightrag.EntityExtractionPromptData) bool
	}{
		{
			name:        "Default values",
			markdownAst: handler.NewMarkdownAst(handler.DocumentConfig{}),
			expected: func(data golightrag.EntityExtractionPromptData) bool {
				return data.Language == "English" &&
					len(data.EntityTypes) > 0 &&
					len(data.Examples) > 0 &&
					data.Goal != ""
			},
		},
		{
			name: "Custom values",
			markdownAst: &handler.MarkdownAst{
				EntityExtractionGoal: "Custom goal",
				EntityTypes:         []string{"person", "place"},
				Language:           "French",
			},
			expected: func(data golightrag.EntityExtractionPromptData) bool {
				return data.Language == "French" &&
					len(data.EntityTypes) == 2 &&
					data.EntityTypes[0] == "person" &&
					data.EntityTypes[1] == "place" &&
					data.Goal == "Custom goal"
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data := tt.markdownAst.EntityExtractionPromptData()
			if !tt.expected(data) {
				t.Errorf("EntityExtractionPromptData() validation failed for %s", tt.name)
				t.Logf("Got: Goal=%s, Language=%s, EntityTypes=%v, Examples=%d",
					data.Goal, data.Language, data.EntityTypes, len(data.Examples))
			}
		})
	}
}

func TestMarkdownAst_ConfigMethods(t *testing.T) {
	tests := []struct {
		name        string
		markdownAst *handler.MarkdownAst
		method      string
		want        interface{}
	}{
		{
			name:        "Default MaxRetries",
			markdownAst: handler.NewMarkdownAst(handler.DocumentConfig{}),
			method:      "MaxRetries",
			want:        0, // Default value
		},
		{
			name: "Custom MaxRetries",
			markdownAst: &handler.MarkdownAst{
				Config: handler.DocumentConfig{MaxRetries: 5},
			},
			method: "MaxRetries",
			want:   5,
		},
		{
			name:        "Default ConcurrencyCount",
			markdownAst: handler.NewMarkdownAst(handler.DocumentConfig{}),
			method:      "ConcurrencyCount",
			want:        1, // Default value
		},
		{
			name: "Custom ConcurrencyCount",
			markdownAst: &handler.MarkdownAst{
				Config: handler.DocumentConfig{ConcurrencyCount: 4},
			},
			method: "ConcurrencyCount",
			want:   4,
		},
		{
			name:        "Default GleanCount",
			markdownAst: handler.NewMarkdownAst(handler.DocumentConfig{}),
			method:      "GleanCount",
			want:        0, // Default value
		},
		{
			name: "Custom GleanCount",
			markdownAst: &handler.MarkdownAst{
				Config: handler.DocumentConfig{GleanCount: 3},
			},
			method: "GleanCount",
			want:   3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got interface{}
			switch tt.method {
			case "MaxRetries":
				got = tt.markdownAst.MaxRetries()
			case "ConcurrencyCount":
				got = tt.markdownAst.ConcurrencyCount()
			case "GleanCount":
				got = tt.markdownAst.GleanCount()
			default:
				t.Fatalf("Unknown method: %s", tt.method)
			}

			if got != tt.want {
				t.Errorf("%s() = %v, want %v", tt.method, got, tt.want)
			}
		})
	}
}

func TestMarkdownAst_InterfaceImplementation(t *testing.T) {
	// This test ensures MarkdownAst correctly implements DocumentHandler interface
	var _ golightrag.DocumentHandler = (*handler.MarkdownAst)(nil)
	
	markdownAst := handler.NewMarkdownAst(handler.DocumentConfig{})
	
	// Test that all methods are callable
	_, err := markdownAst.ChunksDocument("test content")
	if err != nil {
		t.Errorf("ChunksDocument failed: %v", err)
	}
	
	_ = markdownAst.EntityExtractionPromptData()
	_ = markdownAst.MaxRetries()
	_ = markdownAst.ConcurrencyCount()
	_ = markdownAst.BackoffDuration()
	_ = markdownAst.GleanCount()
	_ = markdownAst.MaxSummariesTokenLength()
}

// Helper functions
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