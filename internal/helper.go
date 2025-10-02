package internal

import (
	"fmt"

	"github.com/MegaGrindStone/go-light-rag/llm"
	"github.com/tiktoken-go/tokenizer"
)

func EncodeStringByTokenizers(content, model string) ([]uint, error) {
	// Use default model if not specified
	if model == "" {
		model = "Qwen/Qwen1.5-0.5B"
	}

	// Load tokenizer from HuggingFace model
	tk, err := llm.DownloadTokenizer(model)
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer for model %s: %w", model, err)
	}

	// Encode the content (without special tokens to match tiktoken behavior)
	ids, err := tk.Encode(content)
	if err != nil {
		return nil, fmt.Errorf("failed to encode string: %w", err)
	}

	// Convert []int to []uint
	result := make([]uint, len(ids))
	for i, id := range ids {
		result[i] = uint(id)
	}

	return result, nil
}

// EncodeStringByTiktoken encodes a string into token IDs using the GPT-4o tokenizer.
// It returns a slice of token IDs and an error if tokenization fails.
func EncodeStringByTiktoken(content string) ([]uint, error) {
	enc, err := tokenizer.ForModel(tokenizer.GPT4o)
	if err != nil {
		return nil, fmt.Errorf("failed to get tokenizer: %w", err)
	}

	ids, _, err := enc.Encode(content)
	if err != nil {
		return nil, fmt.Errorf("failed to encode string: %w", err)
	}

	return ids, nil
}

// DecodeTokensByTiktoken decodes token IDs back into a string using the GPT-4o tokenizer.
// It takes a slice of token IDs and returns the decoded string and an error if decoding fails.
func DecodeTokensByTiktoken(tokenIDs []uint) (string, error) {
	enc, err := tokenizer.ForModel(tokenizer.GPT4o)
	if err != nil {
		return "", fmt.Errorf("failed to get tokenizer: %w", err)
	}

	return enc.Decode(tokenIDs)
}

// CountTokens counts the number of tokens in a string using the GPT-4o tokenizer.
// It takes a string input and returns the token count and an error if tokenization fails.
func CountTokens(code string) (int, error) {
	tokenIDs, err := EncodeStringByTiktoken(code)
	if err != nil {
		return 0, fmt.Errorf("failed to encode string: %w", err)
	}
	return len(tokenIDs), nil
}

// CountTokensTokenizers counts the number of tokens in a string using a HuggingFace tokenizer.
// It takes a string input, the model name, and returns the token count and an error if tokenization fails.
func CountTokensTokenizers(code, model string) (int, error) {
	// Use default model if not specified
	if model == "" {
		model = "google-bert/bert-base-uncased"
	}

	tokenIDs, err := EncodeStringByTokenizers(code, model)
	if err != nil {
		return 0, fmt.Errorf("failed to encode string: %w", err)
	}
	return len(tokenIDs), nil
}