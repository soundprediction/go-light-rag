//nolint:lll
package golightrag

type EntityExtractionPromptData struct {
	Goal        string
	EntityTypes []string
	Language    string
	Examples    []EntityExtractionPromptExample

	Input string
}

type EntityExtractionPromptExample struct {
	EntityTypes          []string
	Text                 string
	EntitiesOutputs      []EntityExtractionPromptEntityOutput
	RelationshipsOutputs []EntityExtractionPromptRelationshipOutput
}

type EntityExtractionPromptEntityOutput struct {
	Name        string
	Type        string
	Description string
}

type EntityExtractionPromptRelationshipOutput struct {
	SourceEntity string
	TargetEntity string
	Description  string
	Keywords     []string
	Strength     float64
}

type KeywordExtractionPromptData struct {
	Goal     string
	Examples []KeywordExtractionPromptExample

	Query   string
	History string
}

type KeywordExtractionPromptExample struct {
	Query             string
	LowLevelKeywords  []string
	HighLevelKeywords []string
}

const extractEntitiesPrompt = `---Goal---
{{.Goal}}

---Steps---
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If {{.Language}}, capitalized the name.
- entity_type: One of the following types: [{{range $i, $v := .EntityTypes}}{{if $i}}, {{end}}{{$v}}{{end}}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

3. Return output in {{.Language}} as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
---Examples---
######################
{{- range $i, $example := .Examples}}
Example {{add $i 1}}:

Text:
{{$example.Text}}
################
Output:
  {{- range $output := $example.EntitiesOutputs}}
("entity"<|>"{{$output.Name}}"<|>"{{$output.Type}}"<|>"{{$output.Description}}")##
  {{- end}}
  {{- range $output := $example.RelationshipsOutputs}}
("relationship"<|>"{{$output.SourceEntity}}"<|>"{{$output.TargetEntity}}"<|>"{{$output.Description}}"<|>"{{range $i, $v := $output.Keywords}}{{if $i}}, {{end}}{{$v}}{{end}}"<|>"{{$output.Strength}}")##
  {{- end}}
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

const gleanEntitiesPrompt = `
MANY entities and relationships were missed in the last extraction.

---Remember Steps---

1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, use same language as input text. If {{.Language}}, capitalized the name.
- entity_type: One of the following types: [{{range $i, $v := .EntityTypes}}{{if $i}}, {{end}}{{$v}}{{end}}]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
- relationship_keywords: one or more high-level key words that summarize the overarching nature of the relationship, focusing on concepts or themes rather than specific details
Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_keywords><|><relationship_strength>)

3. Return output in {{.Language}} as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

---Output---

Add them below using the same format:`

const gleanDecideContinuePrompt = `
---Goal---

It appears some entities may have still been missed.

---Output---

Answer ONLY by "YES" OR "NO" if there are still entities that need to be added.`

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
  {{- range $output := $example.EntitiesOutputs}}
("entity"<|>"{{$output.Name}}"<|>"{{$output.Type}}"<|>"{{$output.Description}}")##
  {{- end}}
  {{- range $output := $example.RelationshipsOutputs}}
("relationship"<|>"{{$output.SourceEntity}}"<|>"{{$output.TargetEntity}}"<|>"{{$output.Description}}"<|>"{{range $i, $v := $output.Keywords}}{{if $i}}, {{end}}{{$v}}{{end}}"<|>"{{$output.Strength}}")##
  {{- end}}
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
