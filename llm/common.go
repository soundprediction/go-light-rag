package llm

import (
	"regexp"
	"strings"
)

// RemoveThinkTags removes <think> tags and everything in between them from a string.
func RemoveThinkTags(input string) string {
	re := regexp.MustCompile(`(?s)<think>.*?</think>`)
	return re.ReplaceAllString(input, "")
}

func RemoveMarkdownBackticks(input string) string {
	lines := strings.Split(input, "\n")

	// Filter out lines that start with triple backticks
	var filteredLines []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if !strings.HasPrefix(trimmed, "```") {
			filteredLines = append(filteredLines, line)
		}
	}

	return strings.Join(filteredLines, "\n")
}
