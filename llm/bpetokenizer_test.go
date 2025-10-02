package llm

import (
	"testing"
)

func TestTokenizer_Encode_With_Download(t *testing.T) {
	// This test requires an internet connection to download the tokenizer files.
	modelName := "Qwen/Qwen1.5-0.5B"
	tokenizer, err := DownloadTokenizer(modelName)
	if err != nil {
		t.Fatalf("Failed to download tokenizer: %v", err)
	}

	text := "Hello world! This is the Qwen3 tokenizer.<|endoftext|>"
	t.Logf("Original text: %s", text)

	ids, err := tokenizer.Encode(text)
	if err != nil {
		t.Fatalf("Failed to encode text: %v", err)
	}

	t.Logf("Encoded token IDs: %v", ids)
}
