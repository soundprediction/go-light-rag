package internal

import (
	"fmt"

	"github.com/tiktoken-go/tokenizer"
)

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
