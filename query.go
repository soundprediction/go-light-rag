package golightrag

import "time"

type QueryResult struct {
	GlobalEntities      []EntityContext
	GlobalRelationships []RelationshipContext
	GlobalSources       []SourceContext
	LocalEntities       []EntityContext
	LocalRelationships  []RelationshipContext
	LocalSources        []SourceContext
}

type EntityContext struct {
	Name        string
	Type        string
	Description string
	RefCount    int
	CreatedAt   time.Time
}

type RelationshipContext struct {
	Source      string
	Target      string
	Keywords    string
	Description string
	Weight      float64
	RefCount    int
	CreatedAt   time.Time
}

type SourceContext struct {
	Content  string
	RefCount int
}
