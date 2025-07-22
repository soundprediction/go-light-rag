package golightrag

// EntityExtractionPromptData contains the data needed to generate prompts
// for extracting entities and relationships from text content.
// It includes the goal of extraction, valid entity types, target language,
// example extractions, and the input text to be processed.
type EntityExtractionPromptData struct {
	Goal        string
	EntityTypes []string
	Language    string
	Examples    []EntityExtractionPromptExample

	Input string
}

// EntityExtractionPromptExample provides sample inputs and outputs
// for demonstrating entity extraction to language models.
// It includes sample text content along with the expected entities
// and relationships that should be extracted from the text.
type EntityExtractionPromptExample struct {
	EntityTypes          []string
	Text                 string
	EntitiesOutputs      []EntityExtractionPromptEntityOutput
	RelationshipsOutputs []EntityExtractionPromptRelationshipOutput
}

// EntityExtractionPromptEntityOutput represents the expected output format
// for an entity identified during extraction.
// It includes the entity's name, type, and description.
type EntityExtractionPromptEntityOutput struct {
	Name        string
	Type        string
	Description string
}

// EntityExtractionPromptRelationshipOutput represents the expected output format
// for a relationship identified between entities during extraction.
// It includes source and target entities, description, relevant keywords,
// and a strength value indicating the relationship's importance.
type EntityExtractionPromptRelationshipOutput struct {
	SourceEntity string
	TargetEntity string
	Description  string
	Keywords     []string
	Strength     float64
}

// KeywordExtractionPromptData contains the data needed to generate prompts
// for extracting keywords from user queries and conversation history.
// It includes the goal of keyword extraction, examples for demonstration,
// the current query, and relevant conversation history.
type KeywordExtractionPromptData struct {
	Goal     string
	Examples []KeywordExtractionPromptExample

	Query   string
	History string
}

// KeywordExtractionPromptExample provides sample inputs and outputs
// for demonstrating keyword extraction to language models.
// It includes a sample query along with expected high-level and low-level
// keywords that should be extracted from the query.
type KeywordExtractionPromptExample struct {
	Query             string
	LowLevelKeywords  []string
	HighLevelKeywords []string
}

//nolint:lll
const extractEntitiesPrompt = `---Goal---
{{.Goal}}

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If {{.Language}}, capitalized the name.
- entity_type: STRICTLY use ONLY one of the exact entity types provided here (no variations, plurals, or additions): [{{range $i, $v := .EntityTypes}}{{if $i}}, {{end}}{{$v}}{{end}}]
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity (use a number between 1-10)
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details

3. Extract high-level keywords that summarize the main concepts or themes present in the document.

4. Format your output as a VALID JSON object with the following structure:
{
  "entities": [
    {
      "entity_name": string,
      "entity_type": string (one of the provided entity types ONLY),
      "entity_description": string
    }
  ],
  "relationships": [
    {
      "source_entity": string,
      "target_entity": string,
      "relationship_description": string,
      "relationship_keywords": array of strings,
      "relationship_strength": number (1-10)
    }
  ],
}

5. The JSON output MUST be valid JSON with no explanation text before or after it. Do not include any markdown formatting like backticks, and do not include any text outside the JSON structure.

######################
---Examples---
######################
{{- range $i, $example := .Examples}}
Example {{add $i 1}}:

Text:
{{$example.Text}}
################
Output:
{
  "entities": [
    {{- range $j, $output := $example.EntitiesOutputs}}
    {{if $j}},{{end}}
    {
      "entity_name": "{{$output.Name}}",
      "entity_type": "{{$output.Type}}",
      "entity_description": "{{$output.Description}}"
    }
    {{- end}}
  ],
  "relationships": [
    {{- range $j, $output := $example.RelationshipsOutputs}}
    {{if $j}},{{end}}
    {
      "source_entity": "{{$output.SourceEntity}}",
      "target_entity": "{{$output.TargetEntity}}",
      "relationship_description": "{{$output.Description}}",
      "relationship_keywords": [{{range $k, $v := $output.Keywords}}{{if $k}}, {{end}}"{{$v}}"{{end}}],
      "relationship_strength": {{$output.Strength}}
    }
    {{- end}}
  ],
}
#############################
{{- end}}

#############################
---Real Data---
######################
Entity_types: [{{range $i, $v := .EntityTypes}}{{if $i}}, {{end}}{{$v}}{{end}}]
Text:
{{.Input}}
######################
Output:`

//nolint:lll
const gleanEntitiesPrompt = `
MANY entities and relationships were missed in the last extraction. Please identify additional entities and relationships.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If {{.Language}}, capitalized the name.
- entity_type: STRICTLY use ONLY one of the exact entity types provided here (no variations, plurals, or additions): [{{range $i, $v := .EntityTypes}}{{if $i}}, {{end}}{{$v}}{{end}}]
- entity_description: Comprehensive description of the entity's attributes and activities

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity (use a number between 1-10)
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details

3. Extract additional high-level keywords that summarize concepts or themes that may have been missed in the initial extraction.

4. Format your output as a VALID JSON object with the following structure:
{
  "entities": [
    {
      "entity_name": string,
      "entity_type": string (one of the provided entity types ONLY),
      "entity_description": string
    }
  ],
  "relationships": [
    {
      "source_entity": string,
      "target_entity": string,
      "relationship_description": string,
      "relationship_keywords": array of strings,
      "relationship_strength": number (1-10)
    }
  ],
}

5. The JSON output MUST be valid JSON with no explanation text before or after it. Do not include any markdown formatting like backticks, and do not include any text outside the JSON structure.

---Output---

Please provide the additional entities and relationships in valid JSON format:`

const gleanDecideContinuePrompt = `
---Goal---

It appears some entities may have still been missed.

---Output---

Answer ONLY by "YES" OR "NO" if there are still entities that need to be added.`

//nolint:lll
const summarizeDescriptionsPrompt = `
You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
Given one or two entities, and a list of descriptions, all related to the same entity or group of entities.
Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions.
If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
Make sure it is written in third person, and include the entity names so we the have full context.
Use {{.Language}} as the language.

#######
-Data-
Entities: {{.EntityName}}
Description List: {{.Descriptions}}
#######
Output:
`

//nolint:lll
const keywordExtractionPrompt = `---Role---

You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query and conversation history.

---Goal---

{{.Goal}}

---Instructions---

- Consider both the current query and relevant conversation history when extracting keywords
- Output the keywords in JSON format, it will be parsed by a JSON parser, do not add any extra content in output
- The JSON should have two keys:
  - "high_level_keywords" for overarching concepts or themes
  - "low_level_keywords" for specific entities or details

######################
---Examples---
######################
{{- range $i, $example := .Examples}}
Example {{add $i 1}}:

Query: {{$example.Query}}
################
Output:
\{
  "high_level_keywords": [{{range $i, $v := $example.HighLevelKeywords}}{{if $i}}, {{end}}"{{$v}}"{{end}}],
  "low_level_keywords": [{{range $i, $v := $example.LowLevelKeywords}}{{if $i}}, {{end}}"{{$v}}"{{end}}]
\}
#############################
{{- end}}
-Real Data-
######################
Conversation History:
{{.History}}

Current Query: {{.Query}}
######################
The "Output" should be human text, not unicode characters. Keep the same language as "Query".
Output:

`
