//nolint:lll
package handler

import golightrag "github.com/MegaGrindStone/go-light-rag"

const defaultEntityExtractionGoal = `
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.`

const defaultKeywordExtractionGoal = `
Given the query and conversation history, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.`

var defaultEntityTypes = []string{"organization", "person", "geo", "event", "category"}

var defaultEntityExtractionExamples = []golightrag.EntityExtractionPromptExample{
	{
		EntityTypes: []string{"person", "technology", "mission", "organization", "location"},
		Text: `
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "Alex",
				Type:        "person",
				Description: "Alex is a character who experiences frustration and is observant of the dynamics among other characters.",
			},
			{
				Name:        "Taylor",
				Type:        "person",
				Description: "Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective.",
			},
			{
				Name:        "Jordan",
				Type:        "person",
				Description: "Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device.",
			},
			{
				Name:        "Cruz",
				Type:        "person",
				Description: "Cruz is associated with a vision of control and order, influencing the dynamics among other characters.",
			},
			{
				Name:        "The Device",
				Type:        "technology",
				Description: "The Device is central to the story, with potential game-changing implications, and is revered by Taylor.",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "Alex",
				TargetEntity: "Taylor",
				Description:  "Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device.",
				Keywords:     []string{"power dynamics", "perspective shift"},
				Strength:     7,
			},
			{
				SourceEntity: "Alex",
				TargetEntity: "Jordan",
				Description:  "Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision.",
				Keywords:     []string{"shared goals", "rebellion"},
				Strength:     6,
			},
			{
				SourceEntity: "Taylor",
				TargetEntity: "Jordan",
				Description:  "Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce.",
				Keywords:     []string{"conflict resolution", "mutual respect"},
				Strength:     8,
			},
			{
				SourceEntity: "Jordan",
				TargetEntity: "Cruz",
				Description:  "Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order.",
				Keywords:     []string{"ideological conflict", "rebellion"},
				Strength:     5,
			},
			{
				SourceEntity: "Taylor",
				TargetEntity: "The Device",
				Description:  "Taylor shows reverence towards the device, indicating its importance and potential impact.",
				Keywords:     []string{"reverence", "technological significance"},
				Strength:     9,
			},
		},
	},
	{
		EntityTypes: []string{"company", "index", "commodity", "market_trend", "economic_policy", "biological"},
		Text: `
Stock markets faced a sharp downturn today as tech giants saw significant declines, with the Global Tech Index dropping by 3.4% in midday trading. Analysts attribute the selloff to investor concerns over rising interest rates and regulatory uncertainty.

Among the hardest hit, Nexon Technologies saw its stock plummet by 7.8% after reporting lower-than-expected quarterly earnings. In contrast, Omega Energy posted a modest 2.1% gain, driven by rising oil prices.

Meanwhile, commodity markets reflected a mixed sentiment. Gold futures rose by 1.5%, reaching $2,080 per ounce, as investors sought safe-haven assets. Crude oil prices continued their rally, climbing to $87.60 per barrel, supported by supply constraints and strong demand.

Financial experts are closely watching the Federal Reserve’s next move, as speculation grows over potential rate hikes. The upcoming policy announcement is expected to influence investor confidence and overall market stability.`,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "Global Tech Index",
				Type:        "index",
				Description: "The Global Tech Index tracks the performance of major technology stocks and experienced a 3.4% decline today.",
			},
			{
				Name:        "Nexon Technologies",
				Type:        "company",
				Description: "Nexon Technologies is a tech company that saw its stock decline by 7.8% after disappointing earnings.",
			},
			{
				Name:        "Omega Energy",
				Type:        "company",
				Description: "Omega Energy is an energy company that gained 2.1% in stock value due to rising oil prices.",
			},
			{
				Name:        "Gold Futures",
				Type:        "commodity",
				Description: "Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets.",
			},
			{
				Name:        "Crude Oil",
				Type:        "commodity",
				Description: "Crude oil prices rose to $87.60 per barrel due to supply constraints and strong demand.",
			},
			{
				Name:        "Market Selloff",
				Type:        "market_trend",
				Description: "Market selloff refers to the significant decline in stock values due to investor concerns over interest rates and regulations.",
			},
			{
				Name:        "Federal Reserve Policy Announcement",
				Type:        "economic_policy",
				Description: "The Federal Reserve's upcoming policy announcement is expected to impact investor confidence and market stability.",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "Global Tech Index",
				TargetEntity: "Market Selloff",
				Description:  "The decline in the Global Tech Index is part of the broader market selloff driven by investor concerns.",
				Keywords:     []string{"market performance", "investor sentiment"},
				Strength:     9,
			},
			{
				SourceEntity: "Nexon Technologies",
				TargetEntity: "Global Tech Index",
				Description:  "Nexon Technologies' stock decline contributed to the overall drop in the Global Tech Index.",
				Keywords:     []string{"company impact", "index movement"},
				Strength:     8,
			},
			{
				SourceEntity: "Gold Futures",
				TargetEntity: "Market Selloff",
				Description:  "Gold prices rose as investors sought safe-haven assets during the market selloff.",
				Keywords:     []string{"market reaction", "safe-haven investment"},
				Strength:     10,
			},
			{
				SourceEntity: "Federal Reserve Policy Announcement",
				TargetEntity: "Market Selloff",
				Description:  "Speculation over Federal Reserve policy changes contributed to market volatility and investor selloff.",
				Keywords:     []string{"interest rate impact", "financial regulation"},
				Strength:     7,
			},
		},
	},
	{
		EntityTypes: []string{"economic_policy", "athlete", "event", "location", "record", "organization", "equipment"},
		Text: `
At the World Athletics Championship in Tokyo, Noah Carter broke the 100m sprint record using cutting-edge carbon-fiber spikes.
    `,
		EntitiesOutputs: []golightrag.EntityExtractionPromptEntityOutput{
			{
				Name:        "World Athletics Championship",
				Type:        "event",
				Description: "The World Athletics Championship is a global sports competition featuring top athletes in track and field.",
			},
			{
				Name:        "Tokyo",
				Type:        "location",
				Description: "Tokyo is the host city of the World Athletics Championship.",
			},
			{
				Name:        "Noah Carter",
				Type:        "athlete",
				Description: "Noah Carter is a sprinter who set a new record in the 100m sprint at the World Athletics Championship.",
			},
			{
				Name:        "100m Sprint Record",
				Type:        "record",
				Description: "The 100m sprint record is a benchmark in athletics, recently broken by Noah Carter.",
			},
			{
				Name:        "Carbon-Fiber Spikes",
				Type:        "equipment",
				Description: "Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and traction.",
			},
			{
				Name:        "World Athletics Federation",
				Type:        "organization",
				Description: "The World Athletics Federation is the governing body overseeing the World Athletics Championship and record validations.",
			},
		},
		RelationshipsOutputs: []golightrag.EntityExtractionPromptRelationshipOutput{
			{
				SourceEntity: "World Athletics Championship",
				TargetEntity: "Tokyo",
				Description:  "The World Athletics Championship is being hosted in Tokyo.",
				Keywords:     []string{"event location", "international competition"},
				Strength:     8,
			},
			{
				SourceEntity: "Noah Carter",
				TargetEntity: "100m Sprint Record",
				Description:  "Noah Carter set a new 100m sprint record at the championship.",
				Keywords:     []string{"athlete achievement", "record-breaking"},
				Strength:     10,
			},
			{
				SourceEntity: "Noah Carter",
				TargetEntity: "Carbon-Fiber Spikes",
				Description:  "Noah Carter used carbon-fiber spikes to enhance performance during the race.",
				Keywords:     []string{"athletic equipment", "performance boost"},
				Strength:     7,
			},
			{
				SourceEntity: "World Athletics Federation",
				TargetEntity: "100m Sprint Record",
				Description:  "The World Athletics Federation is responsible for validating and recognizing new sprint records.",
				Keywords:     []string{"sports regulation", "record certification"},
				Strength:     9,
			},
		},
	},
}

var defaultKeywordExtractionExamples = []golightrag.KeywordExtractionPromptExample{
	{
		Query:             "How does international trade influence global economic stability?",
		LowLevelKeywords:  []string{"Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"},
		HighLevelKeywords: []string{"International trade", "Global economic stability", "Economic impact"},
	},
	{
		Query:             "What are the environmental consequences of deforestation on biodiversity?",
		LowLevelKeywords:  []string{"Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"},
		HighLevelKeywords: []string{"Environmental consequences", "Deforestation", "Biodiversity loss"},
	},
	{
		Query:             "What is the role of education in reducing poverty?",
		LowLevelKeywords:  []string{"School access", "Literacy rates", "Job training", "Income inequality"},
		HighLevelKeywords: []string{"Education", "Poverty reduction", "Socioeconomic development"},
	},
}
