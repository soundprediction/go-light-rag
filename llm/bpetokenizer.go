package llm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/dlclark/regexp2"
)

// Pair represents a pair of tokens to be merged.
type Pair struct {
	Left  string
	Right string
}

// Tokenizer is an interface for tokenizing text.

type Tokenizer interface {
	Encode(text string) ([]int, error)
}

// BpeTokenizer holds the vocabulary, merge rules, and special tokens.

type BpeTokenizer struct {
	vocab         map[string]int
	merges        map[Pair]int
	specialTokens map[string]int
	preTokenizeRe *regexp2.Regexp
}

// NewTokenizer creates and initializes a new tokenizer from vocab and merges files.
func NewBpeTokenizer(vocabPath, mergesPath string) (*BpeTokenizer, error) {
	// 1. Load Vocabulary
	vocabFile, err := os.ReadFile(vocabPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read vocab file: %w", err)
	}
	var vocab map[string]int
	if err := json.Unmarshal(vocabFile, &vocab); err != nil {
		return nil, fmt.Errorf("failed to parse vocab JSON: %w", err)
	}

	// 2. Load Merges
	mergesFile, err := os.ReadFile(mergesPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read merges file: %w", err)
	}
	mergesLines := strings.Split(string(mergesFile), "\n")
	merges := make(map[Pair]int)
	// Skip header and empty lines
	for i, line := range mergesLines[1:] {
		if line == "" {
			continue
		}
		parts := strings.Fields(line)
		if len(parts) != 2 {
			continue
		}
		merges[Pair{Left: parts[0], Right: parts[1]}] = i
	}

	// 3. Define Special Tokens (example from Qwen family)
	specialTokens := map[string]int{
		"<|endoftext|>": 151643,
		"<|im_start|>":  151644,
		"<|im_end|>":    151645,
	}

	// 4. Define Pre-tokenization Regex (similar to Qwen3's)
	// This regex splits by categories: letters, numbers, punctuation, and whitespace.
	// It also captures special tokens as a whole.
	specialTokenPattern := `<\|endoftext\|>|<\|im_start\|>|<\|im_end\|>`
	pattern := fmt.Sprintf(`(?i)(%s)|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]+|[^\s\p{L}\p{N}]+`, specialTokenPattern)
	re, err := regexp2.Compile(pattern, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to compile pre-tokenization regex: %w", err)
	}

	return &BpeTokenizer{
		vocab:         vocab,
		merges:        merges,
		specialTokens: specialTokens,
		preTokenizeRe: re,
	}, nil
}

// getPairs finds all adjacent pairs in a sequence of tokens.
func getPairs(tokens []string) map[Pair]bool {
	pairs := make(map[Pair]bool)
	for i := 0; i < len(tokens)-1; i++ {
		pairs[Pair{Left: tokens[i], Right: tokens[i+1]}] = true
	}
	return pairs
}

// bpe performs the Byte Pair Encoding algorithm on a list of tokens.
func (t *BpeTokenizer) bpe(tokens []string) []string {
	if len(tokens) < 2 {
		return tokens
	}

	for {
		pairs := getPairs(tokens)
		if len(pairs) == 0 {
			break
		}

		// Find the merge with the highest priority (lowest rank)
		bestPair := Pair{}
		minRank := int(^uint(0) >> 1) // Max int

		for pair := range pairs {
			if rank, ok := t.merges[pair]; ok {
				if rank < minRank {
					minRank = rank
					bestPair = pair
				}
			}
		}

		// If no mergeable pairs are found in the vocabulary, stop.
		if minRank == int(^uint(0)>>1) {
			break
		}

		// Merge the best pair
		var newTokens []string
		i := 0
		for i < len(tokens) {
			if i < len(tokens)-1 && tokens[i] == bestPair.Left && tokens[i+1] == bestPair.Right {
				newTokens = append(newTokens, bestPair.Left+bestPair.Right)
				i += 2
			} else {
				newTokens = append(newTokens, tokens[i])
				i++
			}
		}
		tokens = newTokens
	}
	return tokens
}

// preTokenize splits the input text into initial chunks.
func (t *BpeTokenizer) preTokenize(text string) []string {
	var parts []string
	match, err := t.preTokenizeRe.FindStringMatch(text)
	for match != nil && err == nil {
		parts = append(parts, match.String())
		match, err = t.preTokenizeRe.FindNextMatch(match)
	}
	return parts
}

// Encode converts a string into a slice of token IDs.
func (t *BpeTokenizer) Encode(text string) ([]int, error) {
	var finalTokenIDs []int

	// Pre-tokenize the input text into chunks
	chunks := t.preTokenize(text)

	for _, chunk := range chunks {
		// If the chunk is a special token, handle it directly.
		if id, isSpecial := t.specialTokens[chunk]; isSpecial {
			finalTokenIDs = append(finalTokenIDs, id)
			continue
		}

		// Convert the chunk to its byte representation, then to a list of initial string tokens.
		// This is the "Byte" part of BPE.
		var initialTokens []string
		for _, b := range []byte(chunk) {
			initialTokens = append(initialTokens, string(rune(b)))
		}

		// Perform BPE merges
		mergedTokens := t.bpe(initialTokens)

		// Convert merged tokens to IDs from the vocabulary
		for _, token := range mergedTokens {
			id, ok := t.vocab[token]
			if !ok {
				// Handle unknown tokens. For simplicity, we return an error.
				// A real implementation might use an <unk> token.
				return nil, fmt.Errorf("token not found in vocabulary: %s", token)
			}
			finalTokenIDs = append(finalTokenIDs, id)
		}
	}

	return finalTokenIDs, nil
}

// DownloadTokenizer downloads the tokenizer files from Hugging Face and returns a new Tokenizer.
func DownloadTokenizer(modelName string) (Tokenizer, error) {
	vocabURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/vocab.json", modelName)
	mergesURL := fmt.Sprintf("https://huggingface.co/%s/resolve/main/merges.txt", modelName)

	vocabPath, err := downloadFile(vocabURL)
	if err != nil {
		return nil, fmt.Errorf("failed to download vocab.json: %w", err)
	}

	mergesPath, err := downloadFile(mergesURL)
	if err != nil {
		return nil, fmt.Errorf("failed to download merges.txt: %w", err)
	}

	return NewBpeTokenizer(vocabPath, mergesPath)
}

func downloadFile(url string) (string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("failed to download file: %s", resp.Status)
	}

	tempFile, err := os.CreateTemp("", "tokenizer-*")
	if err != nil {
		return "", err
	}
	defer tempFile.Close()

	_, err = io.Copy(tempFile, resp.Body)
	if err != nil {
		return "", err
	}

	return tempFile.Name(), nil
}
