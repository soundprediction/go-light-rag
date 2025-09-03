package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

// OpenAICompat provides an implementation of the LLM interface for interacting with OpenAI-compatible API services.
// It manages connections to any OpenAI-compatible server instance and handles chat completions.
type OpenAICompat struct {
	BaseUrl string
	model   string
	params  Parameters

	client             *http.Client
	logger             *slog.Logger
	ChatTemplateKwargs map[string]interface{}
}

// NewOpenAICompat creates a new OpenAICompat instance with the specified host URL and model name.
// The host parameter should be a valid URL pointing to an OpenAI-compatible API server.
func NewOpenAICompat(host, model string, params Parameters, logger *slog.Logger) OpenAICompat {
	return OpenAICompat{
		BaseUrl: strings.TrimSuffix(host, "/"),
		model:   model,
		params:  params,
		client:  &http.Client{Timeout: 110 * time.Second},
		logger:  logger.With(slog.String("module", "openaicompat")),
	}
}

// ChatMessage represents a single message in the conversation
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// ChatCompletionRequest represents the request payload for chat completions
type ChatCompletionRequest struct {
	Model              string         `json:"model"`
	Messages           []ChatMessage  `json:"messages"`
	Temperature        *float32       `json:"temperature,omitempty"`
	TopP               *float32       `json:"top_p,omitempty"`
	Stop               []string       `json:"stop,omitempty"`
	PresencePenalty    *float32       `json:"presence_penalty,omitempty"`
	FrequencyPenalty   *float32       `json:"frequency_penalty,omitempty"`
	Seed               *int           `json:"seed,omitempty"`
	LogitBias          map[string]int `json:"logit_bias,omitempty"`
	Logprobs           *bool          `json:"logprobs,omitempty"`
	TopLogprobs        *int           `json:"top_logprobs,omitempty"`
	MaxTokens          *int           `json:"max_tokens,omitempty"`
	ChatTemplateKwargs map[string]any `json:"chat_template_kwargs,omitempty"`
}

// ChatCompletionResponse represents the response from the chat completion API
type ChatCompletionResponse struct {
	Choices []struct {
		Message ChatMessage `json:"message"`
	} `json:"choices"`
}

// Chat sends a chat message to the OpenAI-compatible API.
func (o OpenAICompat) Chat(messages []string) (string, error) {
	msgs := make([]ChatMessage, len(messages))
	for i, msg := range messages {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs[i] = ChatMessage{
			Role:    role,
			Content: msg,
		}
	}

	req := o.chatRequest(msgs)

	ctx, cancel := context.WithTimeout(context.Background(), 110*time.Second)
	defer cancel()

	resp, err := o.sendRequest(ctx, req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", errors.New("no choices found")
	}

	return resp.Choices[0].Message.Content, nil
}

func (o OpenAICompat) chatRequest(messages []ChatMessage) ChatCompletionRequest {
	req := ChatCompletionRequest{
		Model:    o.model,
		Messages: messages,
		ChatTemplateKwargs: map[string]any{
			"thinking": false,
		},
	}

	if o.params.Temperature != nil {
		req.Temperature = o.params.Temperature
	}
	if o.params.TopP != nil {
		req.TopP = o.params.TopP
	}
	if o.params.Stop != nil {
		req.Stop = o.params.Stop
	}
	if o.params.PresencePenalty != nil {
		req.PresencePenalty = o.params.PresencePenalty
	}
	if o.params.Seed != nil {
		req.Seed = o.params.Seed
	}
	if o.params.FrequencyPenalty != nil {
		req.FrequencyPenalty = o.params.FrequencyPenalty
	}
	if o.params.LogitBias != nil {
		req.LogitBias = o.params.LogitBias
	}
	if o.params.Logprobs != nil {
		req.Logprobs = o.params.Logprobs
	}
	if o.params.TopLogprobs != nil {
		req.TopLogprobs = o.params.TopLogprobs
	}
	if o.params.MaxTokens != nil {
		req.MaxTokens = o.params.MaxTokens
	}

	if o.ChatTemplateKwargs != nil {
		req.ChatTemplateKwargs = o.ChatTemplateKwargs
	}

	return req
}

func (o OpenAICompat) sendRequest(ctx context.Context, req ChatCompletionRequest) (*ChatCompletionResponse, error) {
	jsonData, err := json.Marshal(req)
	// fmt.Printf("jsonData: %s\n", string(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", o.BaseUrl+"/chat/completions", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := o.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("API request failed with status %d: %s", resp.StatusCode, string(body))
	}

	var chatResp ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, fmt.Errorf("error decoding response: %w", err)
	}

	return &chatResp, nil
}
