package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"time"

	goopenai "github.com/sashabaranov/go-openai"
)

// OpenAICompat provides an implementation of the LLM interface for interacting with OpenAI-compatible API services.
// It manages connections to any OpenAI-compatible server instance and handles chat completions.
type OpenAICompat struct {
	BaseUrl string
	model   string
	params  Parameters

	client             *goopenai.Client
	logger             *slog.Logger
	ChatTemplateKwargs map[string]interface{}
}

// NewOpenAICompat creates a new OpenAICompat instance with the specified host URL and model name.
// The host parameter should be a valid URL pointing to an OpenAI-compatible API server.
func NewOpenAICompat(host, apiKey string, model string, params Parameters, logger *slog.Logger) OpenAICompat {
	baseUrl := strings.TrimSuffix(host, "/")

	// Create client configuration with custom base URL
	config := goopenai.DefaultConfig(apiKey)
	config.BaseURL = strings.TrimSuffix(host, "/")
	client := goopenai.NewClientWithConfig(config)

	return OpenAICompat{
		BaseUrl: baseUrl,
		model:   model,
		params:  params,
		client:  client,
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
	Model              string                 `json:"model"`
	Messages           []ChatMessage          `json:"messages"`
	Temperature        *float32               `json:"temperature,omitempty"`
	TopP               *float32               `json:"top_p,omitempty"`
	Stop               []string               `json:"stop,omitempty"`
	PresencePenalty    *float32               `json:"presence_penalty,omitempty"`
	FrequencyPenalty   *float32               `json:"frequency_penalty,omitempty"`
	Seed               *int                   `json:"seed,omitempty"`
	LogitBias          map[string]int         `json:"logit_bias,omitempty"`
	Logprobs           *bool                  `json:"logprobs,omitempty"`
	TopLogprobs        *int                   `json:"top_logprobs,omitempty"`
	MaxTokens          *int                   `json:"max_tokens,omitempty"`
	ChatTemplateKwargs map[string]interface{} `json:"chat_template_kwargs,omitempty"`
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
	// Convert our ChatMessage to goopenai.ChatCompletionMessage
	messages := make([]goopenai.ChatCompletionMessage, len(req.Messages))
	for i, msg := range req.Messages {
		messages[i] = goopenai.ChatCompletionMessage{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	// Create OpenAI request
	openaiReq := goopenai.ChatCompletionRequest{
		Model:    req.Model,
		Messages: messages,
	}

	// Set optional parameters
	if req.Temperature != nil {
		openaiReq.Temperature = *req.Temperature
	}
	if req.TopP != nil {
		openaiReq.TopP = *req.TopP
	}
	if req.Stop != nil {
		openaiReq.Stop = req.Stop
	}
	if req.PresencePenalty != nil {
		openaiReq.PresencePenalty = *req.PresencePenalty
	}
	if req.FrequencyPenalty != nil {
		openaiReq.FrequencyPenalty = *req.FrequencyPenalty
	}
	if req.Seed != nil {
		openaiReq.Seed = req.Seed
	}
	if req.LogitBias != nil {
		openaiReq.LogitBias = req.LogitBias
	}
	if req.Logprobs != nil {
		openaiReq.LogProbs = *req.Logprobs
	}
	if req.TopLogprobs != nil {
		openaiReq.TopLogProbs = *req.TopLogprobs
	}
	if req.MaxTokens != nil {
		openaiReq.MaxTokens = *req.MaxTokens
	}

	// Make the request using the OpenAI client
	resp, err := o.client.CreateChatCompletion(ctx, openaiReq)
	if err != nil {
		return nil, fmt.Errorf("error making request: %w", err)
	}

	// Convert response back to our format
	chatResp := &ChatCompletionResponse{
		Choices: make([]struct {
			Message ChatMessage `json:"message"`
		}, len(resp.Choices)),
	}

	for i, choice := range resp.Choices {
		chatResp.Choices[i].Message = ChatMessage{
			Role:    choice.Message.Role,
			Content: choice.Message.Content,
		}
	}

	return chatResp, nil
}
