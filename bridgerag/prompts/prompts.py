"""
This module centralizes all prompt templates used in the BridgeRAG project.
By managing prompts in a single Python file, we gain several advantages:
1. Performance: Prompts are loaded into memory once at application startup, avoiding
   repeated disk I/O.
2. Centralized Management: All prompts have a single, clear source, making them easy
   to find, modify, and reference.
3. Portability: Prompts are packagedâteau with the Python code, simplifying distribution.
"""

from jinja2 import Template

# Prompt for extracting structured entities and relationships from a text chunk.
# This prompt is designed to be used with a powerful language model.
EXTRACTION_PROMPT_TEMPLATE = Template("""
You are a professional Knowledge Graph builder. Your task is to carefully read the provided text document and extract structured entity and relationship information from it.

**Core Principles**:
1.  **Factuality**: All extracted information must be strictly derived from the provided text. Do not infer or hallucinate information that is not explicitly stated.
2.  **Output Format**: You must strictly return the results in the JSON format defined below. Ensure the output is a **single, complete, and directly parsable JSON object**, without any surrounding ```json ...``` markers or other explanatory text.
3.  **Language**: Use `{{language}}` for the output language.

**Attention: Output Limits & Prioritization**
Your output for a single text chunk is strictly limited to a **maximum of 15 entities and 20 relationship pairs**. If the text contains more information than this, you must prioritize:
-   **For Entities**: Focus on extracting named entities (people, organizations, locations, etc.) and other entities that are central to the main topic of the text. Ignore minor or less relevant entities.
-   **For Relationships**: Focus on extracting the most direct and significant relationships that describe the core events and connections in the text. Ignore trivial or indirect associations.

---
### **Steps & Data Structure Definition**

**Step 1: Identify All Entities**
For each identified entity, extract the following information and place it in the "entities" list:
- `entity_name`: (String) The unique name of the entity.
- `entity_description`: (String) A comprehensive description of the entity's attributes and activities, *based only on information present in the input text*. If the text provides insufficient information, return "No description available in the text."
- `is_named_entity`: (Boolean) Set to `true` if the entity is a specific, proper noun (e.g., a person, organization, location, specific technology). Set to `false` if it is a general concept or category. For example, 'Microsoft Corporation', 'Transformer Architecture', and 'GPT-4 Model' are named entities, whereas 'a company', 'a technology', and 'a model' are not.

**Step 2: Identify Relationships Between Entities**
From the entities identified in Step 1, find all pairs (source_entity, target_entity) that have a clear association and extract the following information into the "relationships" list:

- `source_entity`: (String) The name of the source entity. **This name MUST EXACTLY MATCH one of the `entity_name` values from the "entities" list above.**
- `target_entity`: (String) The name of the target entity. **This name MUST EXACTLY MATCH one of the `entity_name` values from the "entities" list above.**
- `relationship_description`: (String) An explanation of why you believe the source and target entities are related.
- `relationship_strength`: (Integer, 1-10) A numerical score representing the strength of the relationship.
- `relationship_keywords`: (List of Strings) One or more high-level keywords that summarize the general nature of the relationship.

**Step 3: Identify Content Keywords**
Identify high-level keywords that summarize the main concepts, themes, or topics of the entire text and place them in the "content_keywords" list.

---
### **Examples**

**Example 1:**
Text:
```
The stock market saw a sharp decline today as major tech giants experienced significant losses, with the Global Tech Index falling 3.4% in midday trading. Analysts attribute the sell-off to investor concerns over rising interest rates and regulatory uncertainty.

Nexon Technologies, one of the hardest-hit companies, saw its shares plummet by 7.8% after reporting disappointing quarterly earnings. In contrast, Omega Energy managed a modest gain of 2.1%, buoyed by rising oil prices.

Meanwhile, sentiment in the commodities market was mixed. Gold futures rose by 1.5% to $2,080 per ounce as investors sought safe-haven assets. Crude oil continued its upward trend, climbing to $87.60 per barrel, supported by tight supply and robust demand.

Financial experts are closely watching the Federal Reserve's next move, as speculation grows about a potential rate hike. The upcoming policy statement is expected to impact investor confidence and overall market stability.
```
Output:
{
  "entities": [
    {"entity_name": "Global Tech Index", "entity_description": "The Global Tech Index, which tracks the performance of major technology stocks, fell by 3.4% today.", "is_named_entity": true},
    {"entity_name": "Nexon Technologies", "entity_description": "Nexon Technologies is a tech company whose stock fell by 7.8% after reporting disappointing earnings.", "is_named_entity": true},
    {"entity_name": "Omega Energy", "entity_description": "Omega Energy is an energy company that saw a 2.1% increase in its stock price due to rising oil prices.", "is_named_entity": true},
    {"entity_name": "Gold Futures", "entity_description": "Gold futures rose by 1.5%, indicating increased investor interest in safe-haven assets.", "is_named_entity": true},
    {"entity_name": "Crude Oil", "entity_description": "Crude oil prices increased to $87.60 per barrel due to tight supply and strong demand.", "is_named_entity": true},
    {"entity_name": "Market Sell-off", "entity_description": "The market sell-off refers to a significant drop in stock prices driven by investor concerns about interest rates and regulation.", "is_named_entity": false},
    {"entity_name": "Federal Reserve Policy Statement", "entity_description": "The upcoming policy statement from the Federal Reserve is expected to impact investor confidence and market stability.", "is_named_entity": true}
  ],
  "relationships": [
    {"source_entity": "Global Tech Index", "target_entity": "Market Sell-off", "relationship_description": "The decline of the Global Tech Index is part of a broader market sell-off driven by investor concerns.", "relationship_keywords": ["Market Performance", "Investor Sentiment"], "relationship_strength": 9},
    {"source_entity": "Nexon Technologies", "target_entity": "Global Tech Index", "relationship_description": "The stock drop of Nexon Technologies contributed to the overall decline of the Global Tech Index.", "relationship_keywords": ["Company Impact", "Index Movement"], "relationship_strength": 8},
    {"source_entity": "Gold Futures", "target_entity": "Market Sell-off", "relationship_description": "During the market sell-off, gold prices increased as investors sought safe-haven assets.", "relationship_keywords": ["Market Reaction", "Safe-Haven Investment"], "relationship_strength": 10},
    {"source_entity": "Federal Reserve Policy Statement", "target_entity": "Market Sell-off", "relationship_description": "Speculation about the Federal Reserve's policy changes contributed to market volatility and the investor sell-off.", "relationship_keywords": ["Interest Rate Impact", "Financial Regulation"], "relationship_strength": 7}
  ],
  "content_keywords": ["Market Decline", "Investor Sentiment", "Commodities", "Federal Reserve", "Stock Performance"]
}

---
**Example 2:**
Text:
```
At the World Athletics Championships in Tokyo, Noah Carter broke the 100-meter sprint record using cutting-edge carbon-fiber spikes.
```
Output:
{
  "entities": [
    {"entity_name": "World Athletics Championships", "entity_description": "The World Athletics Championships is a global sporting event that brings together top track and field athletes.", "is_named_entity": true},
    {"entity_name": "Tokyo", "entity_description": "Tokyo is the host city for the World Athletics Championships.", "is_named_entity": true},
    {"entity_name": "Noah Carter", "entity_description": "Noah Carter is a sprinter who set a new 100-meter sprint record at the World Athletics Championships.", "is_named_entity": true},
    {"entity_name": "100-meter sprint record", "entity_description": "The 100-meter sprint record is a benchmark in athletics, recently broken by Noah Carter.", "is_named_entity": false},
    {"entity_name": "carbon-fiber spikes", "entity_description": "Carbon-fiber spikes are advanced sprinting shoes that provide enhanced speed and grip.", "is_named_entity": false},
    {"entity_name": "World Athletics Federation", "entity_description": "The World Athletics Federation is the governing body that oversees the World Athletics Championships and ratifies records.", "is_named_entity": true}
  ],
  "relationships": [
    {"source_entity": "World Athletics Championships", "target_entity": "Tokyo", "relationship_description": "The World Athletics Championships are being held in Tokyo.", "relationship_keywords": ["Event Location", "International Competition"], "relationship_strength": 8},
    {"source_entity": "Noah Carter", "target_entity": "100-meter sprint record", "relationship_description": "Noah Carter set a new 100-meter sprint record at the championships.", "relationship_keywords": ["Athlete Achievement", "Record Breaking"], "relationship_strength": 10},
    {"source_entity": "Noah Carter", "target_entity": "carbon-fiber spikes", "relationship_description": "Noah Carter used carbon-fiber spikes to enhance his performance during the race.", "relationship_keywords": ["Sports Equipment", "Performance Enhancement"], "relationship_strength": 7},
    {"source_entity": "World Athletics Federation", "target_entity": "100-meter sprint record", "relationship_description": "The World Athletics Federation is responsible for ratifying and recognizing new sprint records.", "relationship_keywords": ["Sports Regulation", "Record Certification"], "relationship_strength": 9}
  ],
  "content_keywords": ["Athletics", "Sprinting", "Record Breaking", "Sports Technology", "Competition"]
}

---
### **Text to Process**

Now, following all the rules above, please analyze the following text document:
```
{{input_text}}
```
Output:
""")

ENTITY_EXTRACTION_ONLY_PROMPT = Template("""
You are a professional Knowledge Graph builder. Your task is to carefully read the provided text and extract ONLY the entities.

**Core Principles**:
1.  **Factuality**: All extracted information must be strictly derived from the provided text. Do not infer or hallucinate information.
2.  **Output Format**: You must strictly return the results in a JSON format containing a single key "entities". Ensure the output is a **single, complete, and directly parsable JSON object**, without any surrounding ```json ...``` markers or other explanatory text. **Crucially, any double quotes within a string value must be properly escaped with a backslash (e.g., "he said \\"hello\\"").**
3.  **Language**: Use `{{language}}` for the output language.
4.  **Entity Diversity**: Prioritize extracting diverse, information-rich entities. Avoid extracting multiple entities that are highly similar or repetitive.
5.  **Rejection of Generic Entities**: Do NOT extract generic, standalone entities like years (e.g., "1997"), dates, or simple numbers. Only extract them if they are part of a specific, named event (e.g., "The 1997 World Fair").
6.  **Handling of Non-Informative Text**: If the text is nonsensical, consists only of boilerplate (e.g., citations, navigation links), or contains no specific, factual information from which to extract entities, you MUST return a JSON with an empty "entities" list, like this: `{"entities": []}`. Do not attempt to extract entities from meaningless content.

**Attention: Output Limits & Prioritization**
Your output for a single text chunk is strictly limited to a **maximum of 30 entities**. If the text contains more, prioritize named entities (people, organizations, locations) and other entities central to the main topic.

---
### **Data Structure Definition**

For each identified entity, extract the following information and place it in the "entities" list:
- `entity_name`: (String) The unique name of the entity.
- `entity_description`: (String) A comprehensive description based *only* on the input text. If none, return "No description available in the text."
- `is_named_entity`: (Boolean) Set to `true` for proper nouns (e.g., 'Microsoft Corporation'), `false` for general concepts (e.g., 'a company').

---
### **Example**
Text:
```
At the World Athletics Championships in Tokyo, Noah Carter broke the 100-meter sprint record.
```
Output:
{
  "entities": [
    {"entity_name": "World Athletics Championships", "entity_description": "A global sporting event.", "is_named_entity": true},
    {"entity_name": "Tokyo", "entity_description": "The host city for the event.", "is_named_entity": true},
    {"entity_name": "Noah Carter", "entity_description": "A sprinter who set a new record.", "is_named_entity": true},
    {"entity_name": "100-meter sprint record", "entity_description": "A benchmark in athletics.", "is_named_entity": false}
  ]
}

---
### **Text to Process**
Now, following all the rules above, please analyze the following text:
```
{{input_text}}
```
Output:
""")

ENTITY_GLEANING_PROMPT = Template("""
You are a detail-oriented fact-checker. You are reviewing the work of another AI that extracted entities from a text. Your task is to find any entities that the first AI missed.

**Core Principles**:
1.  **Find Omissions**: Your only job is to identify entities present in the original `Text to Process` but absent from the `Previously Extracted Entities` list.
2.  **No Duplicates**: Do not repeat any entities that have already been extracted.
3.  **Strict Format**: Return a JSON object with a single key "entities". If you find no new entities, you MUST return an empty list: `{"entities": []}`.
4.  **Language**: Use `{{language}}` for the output language.

---
### **Context**

**Previously Extracted Entities:**
```json
{{previously_extracted_entities}}
```

**Text to Process:**
```
{{input_text}}
```

---
### **Your Task**
Review the `Text to Process` again. Extract any valid entities that are NOT in the `Previously Extracted Entities` list, following the same data structure as the original extraction.

**Output:**
""")

RELATION_EXTRACTION_PROMPT = Template("""
You are a professional Knowledge Graph builder. Your task is to identify and extract relationships **between a predefined list of entities** based on a given text.

**Core Principles**:
1.  **Factuality**: All relationships must be strictly derived from the provided text.
2.  **Strict Scope**: WARNING: You MUST only identify relationships **between the entities provided in the `List of Entities`**. Do not create relationships involving entities not on this list. Any violation of this rule will be considered a failure.
3.  **Output Format**: You must strictly return a JSON object with a single key "relationships". If no relationships are found, you MUST return an empty list `{"relationships": []}`.
4.  **Language**: Use `{{language}}` for the output language.

**Attention: Output Limits & Prioritization**
Your output is strictly limited to a **maximum of 20 relationship pairs**. Prioritize the most direct and significant relationships.

---
### **Data Structure Definition**

For each identified relationship, extract the following:
- `source_entity`: (String) The name of the source entity. **This name MUST EXACTLY MATCH one of the names from the `List of Entities` below.**
- `target_entity`: (String) The name of the target entity. **This name MUST EXACTLY MATCH one of the names from the `List of Entities` below.**
- `relationship_description`: (String) An explanation of why the entities are related, based on the text.
- `relationship_strength`: (Integer, 1-10) The strength of the relationship.
- `relationship_keywords`: (List of Strings) High-level keywords summarizing the relationship.

---
### **Example**
**Text to Process:**
```
At the World Athletics Championships in Tokyo, Noah Carter broke the 100-meter sprint record. The event is governed by the World Athletics Federation.
```
**List of Entities:**
```json
[
    "World Athletics Championships",
    "Tokyo",
    "Noah Carter",
    "100-meter sprint record",
    "World Athletics Federation"
]
```
**Output:**
{
  "relationships": [
    {"source_entity": "World Athletics Championships", "target_entity": "Tokyo", "relationship_description": "The World Athletics Championships are being held in Tokyo.", "relationship_keywords": ["Event Location"], "relationship_strength": 8},
    {"source_entity": "Noah Carter", "target_entity": "100-meter sprint record", "relationship_description": "Noah Carter set a new 100-meter sprint record.", "relationship_keywords": ["Athlete Achievement"], "relationship_strength": 10},
    {"source_entity": "World Athletics Federation", "target_entity": "World Athletics Championships", "relationship_description": "The World Athletics Federation governs the championships.", "relationship_keywords": ["Governing Body"], "relationship_strength": 9}
  ]
}
---
### **Data to Process**

**List of Entities:**
```json
{{entity_list}}
```

**Text to Process:**
```
{{input_text}}
```

**Output:**
""")

# Prompt for summarizing the aggregated information of a single entity.
# This includes its direct descriptions and all its relationships with other entities.
ENTITY_SUMMARY_PROMPT_TEMPLATE = Template("""
You are a professional Knowledge Analyst. Your task is to synthesize the provided information about a single entity into a concise, well-structured, and neutral summary.

**Core Principles**:
1.  **Synthesis, not Invention**: Your summary must be strictly based on the provided "Direct Descriptions" and "Relations". Do not add any external knowledge.
2.  **Neutral Tone**: The summary should be objective and factual.
3.  **Language**: Use `{{language}}` for the output summary.

---
### **Information Provided for Entity: `{{entity_name}}`**

**Direct Descriptions from Text:**
```
{{direct_descriptions}}
```

**Identified Relations:**
```
{{relations}}
```

---
### **Your Task**

Based on all the information above, generate a single, coherent paragraph that serves as a comprehensive summary for the entity `{{entity_name}}`. The summary should integrate both the direct descriptions and the key facts from the relationships. 

**Summary:**
""")

DOCUMENT_SUMMARY_PROMPT_TEMPLATE = Template("""
You are a highly skilled editor and author, tasked with creating a concise, fluid summary of a document. You will be provided with a list of key named entities and their detailed descriptions, which have been extracted from the original text.

**Core Directives**:
1.  **Mandatory Inclusion**: You MUST skillfully and naturally weave **every single named entity** from the provided list into your summary. Do not omit any.
2.  **Information Grounding**: The summary must be factually grounded in the descriptions provided. Do not introduce outside information or make assumptions beyond what is given.
3.  **Coherent Narrative**: The output should be a single, well-structured paragraph that reads like a professional summary, not just a list of facts. Connect the entities into a coherent narrative.
4.  **Language**: Use `{{language}}` for the output summary.

---
### **Key Named Entities and Descriptions from the Document**

{{entity_list}}

---
### **Your Task**

Based on *only* on the information above, generate a single, coherent paragraph that summarizes the original document. Ensure all entities are mentioned and the total length is within the 500-token limit.

**Document Summary:**
""") 

# Prompt for assessing the similarity of two entities based on their descriptions.
# This is used in the entity linking step to determine if two entities from different
# documents refer to the same real-world object.
ENTITY_SIMILARITY_PROMPT = Template("""
You are a highly intelligent entity linking expert. Your task is to determine if two entities, described below, refer to the same real-world person, organization, place, or concept.

**Core Principles**:
1.  **Analyze Descriptions**: Base your judgment solely on the provided descriptions for Entity 1 and Entity 2.
2.  **Context is Key**: Consider the context provided in the descriptions. The same name can refer to different things in different contexts.
3.  **Strict JSON Output**: You must return your analysis as a single, directly parsable JSON object with two keys: "reasoning" and "score". Do not add any extra text.

---
### **Entity Descriptions to Compare**

**Entity 1 Name:** `{{entity_1_name}}`
**Entity 1 Description:**
```
{{entity_1_description}}
```

**Entity 2 Name:** `{{entity_2_name}}`
**Entity 2 Description:**
```
{{entity_2_description}}
```

---
### **Your Task**

Based on the descriptions, provide a similarity score from 1 (completely different) to 10 (definitely the same). Provide a brief reasoning for your score.

**Output Format:**
```json
{
  "reasoning": "A brief explanation of your decision-making process.",
  "score": <an integer between 1 and 10>
}
```

**JSON Output:**
""") 

# Prompt for extracting named entities from a user's question for routing purposes.
QUESTION_ENTITY_EXTRACTION_PROMPT = Template("""
You are an AI assistant specializing in Natural Language Understanding and Information Retrieval. Your task is to extract the key named entities from the user's question.

**Core Principles**:
1.  **Identify Named Entities**: Focus on proper nouns such as people, organizations, locations, specific products, creative works (e.g., movies, books), or technical terms that are central to the user's query.
2.  **Ignore General Concepts**: Do not extract common nouns or general concepts. For example, in "What is the impact of AI on the economy?", extract "AI", but not "impact" or "economy".
3.  **Strict JSON Output**: Your output must be a single, directly parsable JSON object in the format `{"entities": ["entity1", "entity2", ...]}`. Do not include any other text, explanations, or markdown. If no named entities are found, return an empty list: `{"entities": []}`.

---
### **Examples**

**Question:** "What were the key findings of the AlphaGo project by DeepMind, and how did it affect the game of Go?"
**Output:**
{
  "entities": ["AlphaGo", "DeepMind"]
}

**Question:** "Who is the director of the movie The Matrix?"
**Output:**
{
  "entities": ["The Matrix"]
}

**Question:** "Tell me about the latest financial results of Microsoft and its collaboration with OpenAI."
**Output:**
{
  "entities": ["Microsoft", "OpenAI"]
}

**Question:** "How does blockchain technology work?"
**Output:**
{
  "entities": ["blockchain"]
}


---
### **User Question to Process**

**Question:** "{{user_question}}"

**JSON Output:**
""") 

# Prompt for generating a retrieval plan based on candidate documents.
RETRIEVAL_PLAN_PROMPT = Template("""
You are a master AI research assistant. Your task is to analyze a user's question and a list of candidate documents to create a strategic retrieval plan. This plan will guide a retrieval system to gather the most relevant information for answering the question.

**Core Principles**:
1.  **Analyze Relevance**: Carefully evaluate each document's summary and its list of named entities to determine how relevant it is to the user's question.
2.  **Strategic Division**: Divide the documents into two categories:
    -   `main_documents`: A small set of 1-3 documents that are most central and directly relevant to answering the core of the question. These will be analyzed in-depth.
    -   `assist_documents`: Other documents that contain useful, but secondary, information (e.g., specific entities, background context). For these, we only need to look up specific entities.
3.  **Entity Selection**: For `assist_documents`, you MUST identify the specific named entities from that document's entity list that are relevant to the user's question. This is crucial for focused retrieval.
4.  **Strict JSON Output**: Your output must be a single, directly parsable JSON object in the specified format. Do not include any other text, explanations, or markdown.

---
### **Input Information**

**User Question:** 
```
{{user_question}}
```

**Candidate Documents:**
```
{{candidate_documents_context}}
```

---
### **Your Task & Output Format**

Based on the user question and the provided documents, generate a JSON object with the following structure:
-   `reasoning`: (String) A brief, step-by-step thought process on how you formulated the plan.
-   `main_documents`: (List of Strings) A list of `doc_id`s for the main documents.
-   `assist_documents`: (Object) An object where each key is a `doc_id` and the value is a list of specific, relevant entity names to retrieve from that document.

**Example Output Format:**
```json
{
  "reasoning": "The user is asking about the relationship between AIGC and Digital Humans. Doc_A provides a direct overview, making it a main document. Doc_B focuses on AIGC applications, so I will retrieve the 'Digital Human' entity from it for context. Doc_C details the history of 'Virtual Avatars', which is a related concept worth checking.",
  "main_documents": ["Doc_A"],
  "assist_documents": {
    "Doc_B": ["Digital Human"],
    "Doc_C": ["Virtual Avatars"]
  }
}
```

**JSON Output:**
""") 

ENTITY_SELECTION_PROMPT = Template("""
# ROLE: You are an expert assistant for knowledge graph question answering. Your task is to select the most relevant entities from a provided list that are needed to answer a given question.

# CONTEXT:
You will receive the following information:
- A user's `question`.
- A `document_summary` from a document that is highly relevant to the question.
- A `list_of_entities` extracted from that same document.

# INSTRUCTIONS:
1.  Carefully analyze the `question` to understand the user's intent.
2.  Use the `document_summary` to understand the main topic and context of the document.
3.  Review the `list_of_entities` and select ONLY those entities that are directly necessary to formulate a comprehensive answer to the `question`.
4.  Do NOT invent new entities that are not in the provided list.
5.  Do NOT select entities that are only tangentially related or not mentioned in the question.
6.  If no entities from the list are relevant to the question, you MUST return an empty list.

# OUTPUT FORMAT:
Your output MUST be a valid JSON object with a single key "selected_entities". The value should be a list of strings, where each string is an entity name you selected.

# EXAMPLE:
## CONTEXT:
- question: "What is the relationship between DeepMind's AlphaGo and the game of Go?"
- document_summary: "This document describes the development of AlphaGo, a computer program by Google DeepMind, and its historic match against Lee Sedol, a Go champion. It explains the neural network architecture and reinforcement learning techniques used."
- list_of_entities: ["AlphaGo", "Google DeepMind", "Lee Sedol", "neural network", "reinforcement learning", "computer program", "game of Go", "match"]

## YOUR OUTPUT:
```json
{
    "selected_entities": ["AlphaGo", "Google DeepMind", "game of Go", "Lee Sedol"]
}
```

# BEGIN!
## CONTEXT:
- question: "{{ question }}"
- document_summary: "{{ doc_summary }}"
- list_of_entities: {{ entities_list_str }}

## YOUR OUTPUT:
""")


SYNTHESIS_PROMPT = Template("""
# ROLE: You are an advanced reasoning agent. Your primary goal is to provide clear, concise, and factual answers based *only* on the information provided. You will analyze context, decide if it's sufficient, and then either answer directly or ask a targeted sub-question to gather more facts.

# CONTEXT:
You are given a `QUESTION` and a `CONTEXT` section containing retrieved information. You may also receive a `HISTORY` of previous questions and summaries from prior reasoning steps.

## CONTEXT:
{{ evidence_list }}

## HISTORY:
{{ history }}

# TASK:
Your goal is to answer the `QUESTION`. To do this, you must first make a critical decision:

1.  **Analyze Sufficiency**: Is the information in `CONTEXT` and `HISTORY` sufficient to formulate a complete and accurate answer to the `QUESTION`?

2.  **Make a Decision**:
    *   **If YES (information is sufficient)**: Your decision is "ANSWER". You will then synthesize the information to construct a final, comprehensive answer. Your answer must be direct, concise, and strictly based on the provided information. **Be as brief as possible. If the answer is a name, number, or short phrase, provide just that—do not wrap it in a full sentence. Avoid conversational phrases like "The answer is...".**
    *   **If NO (information is insufficient)**: Your decision is "SUB_QUESTION". You must then:
        a.  Formulate a new, more specific sub-question that targets the missing information. This sub-question will be used to perform another round of retrieval.
        b.  Write a concise summary of the knowledge you *have* gathered so far from the context and history. This summary will be passed to the next reasoning step.

# OUTPUT FORMAT:
Your output MUST be a valid JSON object. It must contain three keys: `decision`, `content`, and `summary`.

*   `"decision"`: (string) Must be either "ANSWER" or "SUB_QUESTION".
*   `"content"`: (string)
    *   If `decision` is "ANSWER", this is your final, synthesized answer to the user's question.
    *   If `decision` is "SUB_QUESTION", this is the new sub-question you formulated.
*   `"summary"`: (string) A brief summary of the key findings from the current context, which will serve as history for the next step.

# EXAMPLE 1: Sufficient Information (Direct Answer Style)
## INPUT:
- QUESTION: "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?"
- CONTEXT: "Shirley Temple, an actress known for her role as Corliss Archer in the film 'Kiss and Tell', was appointed to the role of United States Chief of Protocol in 1976."
- HISTORY: "None."

## YOUR OUTPUT:
```json
{
    "decision": "ANSWER",
    "content": "Chief of Protocol",
    "summary": "Shirley Temple, the actress from 'Kiss and Tell', held the position of Chief of Protocol."
}
```

# EXAMPLE 2: Insufficient Information
## INPUT:
- QUESTION: "What was the economic impact of the Apollo program?"
- CONTEXT: "The Apollo program was a series of human spaceflight missions undertaken by NASA... It culminated in the first crewed lunar landing in 1969."
- HISTORY: "Initial query about the Apollo program."

## YOUR OUTPUT:
```json
{
    "decision": "SUB_QUESTION",
    "content": "What were the specific technological innovations and commercial spin-offs resulting from the Apollo program?",
    "summary": "The Apollo program was a NASA initiative for human spaceflight, which achieved the first moon landing in 1969. Specific economic details are not yet available."
}
```

# BEGIN!

## QUESTION:
{{ user_question }}

## YOUR OUTPUT:
"""
)


PROMPTS = {
    "extraction": EXTRACTION_PROMPT_TEMPLATE,
    "entity_extraction_only": ENTITY_EXTRACTION_ONLY_PROMPT,
    "entity_gleaning": ENTITY_GLEANING_PROMPT,
    "relation_extraction": RELATION_EXTRACTION_PROMPT,
    "entity_summary": ENTITY_SUMMARY_PROMPT_TEMPLATE,
    "document_summary": DOCUMENT_SUMMARY_PROMPT_TEMPLATE,
    "entity_similarity": ENTITY_SIMILARITY_PROMPT,
    "question_entity_extraction": QUESTION_ENTITY_EXTRACTION_PROMPT,
    "retrieval_plan": RETRIEVAL_PLAN_PROMPT,
    "entity_selection": ENTITY_SELECTION_PROMPT,
    "synthesis": SYNTHESIS_PROMPT
}