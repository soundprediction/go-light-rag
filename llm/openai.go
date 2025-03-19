package llm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"time"

	goopenai "github.com/sashabaranov/go-openai"
)

// OpenAI provides an implementation of the LLM interface for interacting with OpenAI's language models.
type OpenAI struct {
	model  string
	params Parameters

	client *goopenai.Client
	logger *slog.Logger
}

// NewOpenAI creates a new OpenAI instance.
func NewOpenAI(apiKey, model string, params Parameters, logger *slog.Logger) OpenAI {
	return OpenAI{
		model:  model,
		params: params,
		client: goopenai.NewClient(apiKey),
		logger: logger.With(slog.String("module", "openai")),
	}
}

// Chat sends a chat message to the OpenAI API.
func (o OpenAI) Chat(messages []string) (string, error) {
	msgs := make([]goopenai.ChatCompletionMessage, len(messages))
	for i, msg := range messages {
		role := goopenai.ChatMessageRoleUser
		if i%2 == 1 {
			role = goopenai.ChatMessageRoleAssistant
		}
		msgs[i] = goopenai.ChatCompletionMessage{
			Role:    role,
			Content: msg,
		}
	}

	req := o.chatRequest(msgs)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	resp, err := o.client.CreateChatCompletion(ctx, req)
	if err != nil {
		return "", fmt.Errorf("error sending request: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", errors.New("no choices found")
	}

	return resp.Choices[0].Message.Content, nil
}

func (o OpenAI) chatRequest(messages []goopenai.ChatCompletionMessage) goopenai.ChatCompletionRequest {
	req := goopenai.ChatCompletionRequest{
		Model:    o.model,
		Messages: messages,
	}

	if o.params.Temperature != nil {
		req.Temperature = *o.params.Temperature
	}
	if o.params.TopP != nil {
		req.TopP = *o.params.TopP
	}
	if o.params.Stop != nil {
		req.Stop = o.params.Stop
	}
	if o.params.PresencePenalty != nil {
		req.PresencePenalty = *o.params.PresencePenalty
	}
	if o.params.Seed != nil {
		req.Seed = o.params.Seed
	}
	if o.params.FrequencyPenalty != nil {
		req.FrequencyPenalty = *o.params.FrequencyPenalty
	}
	if o.params.LogitBias != nil {
		req.LogitBias = o.params.LogitBias
	}
	if o.params.Logprobs != nil {
		req.LogProbs = *o.params.Logprobs
	}
	if o.params.TopLogprobs != nil {
		req.TopLogProbs = *o.params.TopLogprobs
	}

	return req
}
