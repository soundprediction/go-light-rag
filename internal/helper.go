package internal

import (
	"fmt"

	"github.com/tiktoken-go/tokenizer"
)

func EncodeStringByTiktoken(content string) ([]uint, error) {
	enc, err := tokenizer.ForModel(tokenizer.GPT4o)
	if err != nil {
		return nil, fmt.Errorf("failed to get tokenizer: %v", err)
	}

	ids, _, err := enc.Encode(content)
	if err != nil {
		return nil, fmt.Errorf("failed to encode string: %v", err)
	}

	return ids, nil
}

func DecodeTokensByTiktoken(tokenIDs []uint) (string, error) {
	enc, err := tokenizer.ForModel(tokenizer.GPT4o)
	if err != nil {
		return "", fmt.Errorf("failed to get tokenizer: %v", err)
	}

	return enc.Decode(tokenIDs)
}
