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
	"time"
)

// OpenRouter provides an implementation of the LLM interface for interacting with OpenRouter's language models.
type OpenRouter struct {
	apiKey string
	model  string

	params Parameters

	client *http.Client
	logger *slog.Logger
}

type openRouterMessage struct {
	Role    string `json:"role"`
	Content string `json:"content,omitempty"`
}

type openRouterChatRequest struct {
	Model    string              `json:"model"`
	Messages []openRouterMessage `json:"messages"`

	Temperature       *float32       `json:"temperature,omitempty"`
	TopP              *float32       `json:"top_p,omitempty"`
	TopK              *int           `json:"top_k,omitempty"`
	FrequencyPenalty  *float32       `json:"frequency_penalty,omitempty"`
	PresencePenalty   *float32       `json:"presence_penalty,omitempty"`
	RepetitionPenalty *float32       `json:"repetition_penalty,omitempty"`
	MinP              *float32       `json:"min_p,omitempty"`
	TopA              *float32       `json:"top_a,omitempty"`
	Seed              *int           `json:"seed,omitempty"`
	MaxTokens         *int           `json:"max_tokens,omitempty"`
	LogitBias         map[string]int `json:"logit_bias,omitempty"`
	Logprobs          *bool          `json:"logprobs,omitempty"`
	TopLogprobs       *int           `json:"top_logprobs,omitempty"`
	Stop              []string       `json:"stop,omitempty"`
	IncludeReasoning  *bool          `json:"include_reasoning,omitempty"`
}

type openRouterResponse struct {
	Choices []openRouterChoice `json:"choices"`
}

type openRouterChoice struct {
	Message openRouterMessage `json:"message"`
}

const (
	openRouterAPIEndpoint = "https://openrouter.ai/api/v1"
)

// NewOpenRouter creates a new OpenRouter instance.
func NewOpenRouter(apiKey, model string, params Parameters, logger *slog.Logger) OpenRouter {
	return OpenRouter{
		apiKey: apiKey,
		model:  model,
		params: params,
		client: &http.Client{},
		logger: logger.With(slog.String("module", "openrouter")),
	}
}

// Chat sends a chat message to the OpenRouter API.
func (o OpenRouter) Chat(messages []string) (string, error) {
	msgs := make([]openRouterMessage, len(messages))
	for i, msg := range messages {
		role := "user"
		if i%2 == 1 {
			role = "assistant"
		}
		msgs[i] = openRouterMessage{
			Role:    role,
			Content: msg,
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	resp, err := o.doRequest(ctx, msgs)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("unexpected status code: %d, body: %s", resp.StatusCode, string(body))
	}

	var res openRouterResponse
	if err := json.NewDecoder(resp.Body).Decode(&res); err != nil {
		return "", fmt.Errorf("error decoding response: %w", err)
	}

	if len(res.Choices) == 0 {
		return "", errors.New("no choices found")
	}

	return res.Choices[0].Message.Content, nil
}

func (o OpenRouter) doRequest(ctx context.Context, messages []openRouterMessage) (*http.Response, error) {
	reqBody := openRouterChatRequest{
		Model:    o.model,
		Messages: messages,

		Temperature:       o.params.Temperature,
		TopP:              o.params.TopP,
		TopK:              o.params.TopK,
		FrequencyPenalty:  o.params.FrequencyPenalty,
		PresencePenalty:   o.params.PresencePenalty,
		RepetitionPenalty: o.params.RepetitionPenalty,
		MinP:              o.params.MinP,
		TopA:              o.params.TopA,
		Seed:              o.params.Seed,
		MaxTokens:         o.params.MaxTokens,
		LogitBias:         o.params.LogitBias,
		Logprobs:          o.params.Logprobs,
		TopLogprobs:       o.params.TopLogprobs,
		Stop:              o.params.Stop,
		IncludeReasoning:  o.params.IncludeReasoning,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("error marshaling request: %w", err)
	}

	o.logger.Debug("Request Body", slog.String("body", string(jsonBody)))

	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		openRouterAPIEndpoint+"/chat/completions", bytes.NewBuffer(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("error creating request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+o.apiKey)
	req.Header.Set("HTTP-Referer", "https://github.com/MegaGrindStone/mcp-web-ui/")
	req.Header.Set("X-Title", "MCP Web UI")

	resp, err := o.client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("unexpected status code: %d, body: %s, request: %s", resp.StatusCode, string(body), jsonBody)
	}

	return resp, nil
}
