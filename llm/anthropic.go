package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	goopenai "github.com/sashabaranov/go-openai"
)

// Anthropic provides an interface to the Anthropic API for large language model interactions. It implements
// the LLM interface and handles streaming chat completions using Claude models.
type Anthropic struct {
	apiKey    string
	model     string
	maxTokens int

	params Parameters

	client *http.Client
}

type anthropicMessage struct {
	Role    string                    `json:"role"`
	Content []anthropicMessageContent `json:"content"`
}

type anthropicMessageContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type anthropicChatRequest struct {
	Model     string             `json:"model"`
	Messages  []anthropicMessage `json:"messages"`
	MaxTokens int                `json:"max_tokens"`

	StopSequences []string `json:"stop_sequences,omitempty"`
	Temperature   *float32 `json:"temperature,omitempty"`
	TopK          *int     `json:"top_k,omitempty"`
	TopP          *float32 `json:"top_p,omitempty"`
}

const (
	anthropicAPIEndpoint = "https://api.anthropic.com/v1"
)

// NewAnthropic creates a new Anthropic instance with the specified API key, model name, and maximum
// token limit. It initializes an HTTP client for API communication and returns a configured Anthropic
// instance ready for chat interactions.
func NewAnthropic(apiKey, model string, maxTokens int, params Parameters) Anthropic {
	return Anthropic{
		apiKey:    apiKey,
		model:     model,
		maxTokens: maxTokens,
		params:    params,
		client:    &http.Client{},
	}
}

// Chat sends a chat message to the Anthropic API.
func (a Anthropic) Chat(messages []string) (string, error) {
	msgs := make([]anthropicMessage, len(messages))
	for i, msg := range messages {
		role := goopenai.ChatMessageRoleUser
		if i%2 == 1 {
			role = goopenai.ChatMessageRoleAssistant
		}
		msgs[i] = anthropicMessage{
			Role:    role,
			Content: []anthropicMessageContent{{Type: "text", Text: msg}},
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	resp, err := a.doRequest(ctx, msgs)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(body))
	}

	var msg anthropicMessage
	if err := json.NewDecoder(resp.Body).Decode(&msg); err != nil {
		return "", fmt.Errorf("error decoding response: %w", err)
	}

	if len(msg.Content) == 0 {
		return "", fmt.Errorf("empty response content")
	}

	return msg.Content[0].Text, nil
}

func (a Anthropic) doRequest(ctx context.Context, messages []anthropicMessage) (*http.Response, error) {
	reqBody := anthropicChatRequest{
		Model:     a.model,
		Messages:  messages,
		MaxTokens: a.maxTokens,

		StopSequences: a.params.Stop,
		Temperature:   a.params.Temperature,
		TopK:          a.params.TopK,
		TopP:          a.params.TopP,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		anthropicAPIEndpoint+"/messages", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", a.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := a.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d, body: %s, request: %s", resp.StatusCode, string(body), jsonBody)
	}

	return resp, nil
}
