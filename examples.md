# Generative AI Usage Examples

This document provides practical examples of how to use our Generative AI framework for common tasks and scenarios.

## Table of Contents

1. [Basic LLM Interaction](#basic-llm-interaction)
2. [Prompt Templates](#prompt-templates)
3. [Retrieval-Augmented Generation](#retrieval-augmented-generation)
4. [Conversation Management](#conversation-management)
5. [Structured Output Generation](#structured-output-generation)
6. [Model Switching](#model-switching)
7. [Batch Processing](#batch-processing)
8. [Evaluation Examples](#evaluation-examples)

## Basic LLM Interaction

### Simple Text Generation

```python
from src.llm.client import LLMClient

# Initialize the client using configuration
client = LLMClient.from_config("config/model.yaml")

# Generate a response
response = client.generate("Explain quantum computing in simple terms.")

print(response.text)
```

### Streaming Response

```python
from src.llm.client import LLMClient

client = LLMClient.from_config("config/model.yaml")

# Stream the response
for chunk in client.generate_stream("Write a short poem about technology."):
    print(chunk.text, end="", flush=True)
```

## Prompt Templates

### Using a Basic Template

```python
from src.prompt_engineering.templates import PromptTemplate

# Define a template
summary_template = PromptTemplate(
    template="Summarize the following text in {word_count} words:\n\n{text}",
    input_variables=["text", "word_count"]
)

# Format the template
formatted_prompt = summary_template.format(
    text="Lorem ipsum dolor sit amet, consectetur adipiscing elit...",
    word_count=100
)

# Use with LLM client
client = LLMClient.from_config("config/model.yaml")
response = client.generate(formatted_prompt)
```

### Creating a Custom Template Class

```python
from src.prompt_engineering.templates import BasePromptTemplate

class ProductDescriptionTemplate(BasePromptTemplate):
    """Generate product descriptions based on attributes."""
    
    def format(self, product_name, features, target_audience, tone="professional"):
        features_text = "\n".join([f"- {feature}" for feature in features])
        
        return f"""
        Generate a compelling product description for {product_name}.
        
        Product features:
        {features_text}
        
        Target audience: {target_audience}
        Tone: {tone}
        
        The description should highlight the key benefits and include a call to action.
        """

# Usage
template = ProductDescriptionTemplate()
prompt = template.format(
    product_name="EcoBoost Water Bottle",
    features=["BPA-free plastic", "Insulated design", "500ml capacity"],
    target_audience="fitness enthusiasts",
    tone="energetic"
)

client = LLMClient.from_config("config/model.yaml")
response = client.generate(prompt)
```

## Retrieval-Augmented Generation

### Setting Up the Vector Database

```python
from src.data.embeddings import EmbeddingGenerator
from src.data.vector_db import VectorDatabase

# Initialize the embedding generator
embedding_generator = EmbeddingGenerator.from_config("config/embeddings.yaml")

# Initialize the vector database
vector_db = VectorDatabase.from_config("config/vector_db.yaml")

# Add documents to the database
documents = [
    "Artificial intelligence is the simulation of human intelligence by machines.",
    "Machine learning is a subset of AI focused on data and algorithms.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing helps computers understand human language."
]

# Generate embeddings and store in the database
for doc in documents:
    embedding = embedding_generator.generate(doc)
    vector_db.add(doc, embedding)
```

### Performing RAG

```python
from src.llm.client import LLMClient
from src.data.embeddings import EmbeddingGenerator
from src.data.vector_db import VectorDatabase
from src.prompt_engineering.templates import PromptTemplate

# Initialize components
client = LLMClient.from_config("config/model.yaml")
embedding_generator = EmbeddingGenerator.from_config("config/embeddings.yaml")
vector_db = VectorDatabase.from_config("config/vector_db.yaml")

# Define RAG template
rag_template = PromptTemplate(
    template="Answer the question based on the following context:\n\nContext: {context}\n\nQuestion: {question}",
    input_variables=["context", "question"]
)

# User question
question = "What is the difference between deep learning and machine learning?"

# Generate embedding for the question
question_embedding = embedding_generator.generate(question)

# Retrieve relevant documents
relevant_docs = vector_db.search(question_embedding, top_k=3)
context = "\n".join(relevant_docs)

# Format prompt with retrieved context
prompt = rag_template.format(context=context, question=question)

# Generate response
response = client.generate(prompt)
print(response.text)
```

## Conversation Management

### Managing a Multi-turn Conversation

```python
from src.llm.client import LLMClient
from src.prompt_engineering.conversation import ConversationManager

# Initialize components
client = LLMClient.from_config("config/model.yaml")
conversation = ConversationManager(
    system_message="You are a helpful assistant that specializes in technology."
)

# First user message
user_message = "What is cloud computing?"
conversation.add_user_message(user_message)
response = client.generate(conversation.get_prompt())
conversation.add_assistant_message(response.text)
print(f"Assistant: {response.text}")

# Second user message
user_message = "What are the main providers?"
conversation.add_user_message(user_message)
response = client.generate(conversation.get_prompt())
conversation.add_assistant_message(response.text)
print(f"Assistant: {response.text}")

# Save conversation for later
conversation.save("data/conversations/cloud_computing_convo.json")
```

### Loading and Continuing a Conversation

```python
from src.llm.client import LLMClient
from src.prompt_engineering.conversation import ConversationManager

# Initialize client
client = LLMClient.from_config("config/model.yaml")

# Load existing conversation
conversation = ConversationManager.load("data/conversations/cloud_computing_convo.json")

# Continue the conversation
user_message = "Which one is best for a small startup?"
conversation.add_user_message(user_message)
response = client.generate(conversation.get_prompt())
conversation.add_assistant_message(response.text)
print(f"Assistant: {response.text}")
```

## Structured Output Generation

### Generating JSON Responses

```python
from src.llm.client import LLMClient
from src.prompt_engineering.templates import PromptTemplate

# Initialize client
client = LLMClient.from_config("config/model.yaml")

# Define a template for structured output
json_template = PromptTemplate(
    template="""
    Extract the following information from the text and return it as JSON:
    - Person names
    - Organizations
    - Locations
    
    Text: {text}
    
    Return the information in the following JSON format:
    {{
        "people": ["name1", "name2", ...],
        "organizations": ["org1", "org2", ...],
        "locations": ["location1", "location2", ...]
    }}
    """,
    input_variables=["text"]
)

# Sample text
text = "John Smith from Microsoft met with Sarah Johnson at Google's headquarters in Mountain View to discuss a partnership with Apple."

# Format prompt
prompt = json_template.format(text=text)

# Generate structured response
response = client.generate(prompt)

# Parse JSON response
import json
structured_data = json.loads(response.text)
print(structured_data)
```

### Using Output Parsers

```python
from src.llm.client import LLMClient
from src.prompt_engineering.templates import PromptTemplate
from src.prompt_engineering.output_parsers import JSONOutputParser

# Initialize components
client = LLMClient.from_config("config/model.yaml")
parser = JSONOutputParser()

# Define schema
schema = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "summary": {"type": "string"},
        "key_points": {"type": "array", "items": {"type": "string"}},
        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]}
    }
}

# Create template with parser instructions
template = PromptTemplate(
    template="""
    Analyze the following article and extract the requested information:
    
    Article: {article}
    
    {format_instructions}
    """,
    input_variables=["article"],
    partial_variables={"format_instructions": parser.get_format_instructions(schema)}
)

# Sample article
article = "In a groundbreaking announcement, researchers at MIT have developed a new battery technology that could double the life of electric vehicles. The innovation uses sustainable materials and could reduce production costs by 30%. Industry experts are calling this a game-changer for the EV market."

# Format prompt
prompt = template.format(article=article)

# Generate and parse response
response = client.generate(prompt)
parsed_output = parser.parse(response.text)
print(parsed_output)
```

## Model Switching

### Dynamically Selecting Models

```python
from src.llm.client import LLMClientFactory

# Initialize factory
client_factory = LLMClientFactory()

# Function to select appropriate model based on task
def get_client_for_task(task_type, complexity):
    if task_type == "creative" and complexity == "high":
        return client_factory.get_client("gpt-4")
    elif task_type == "factual" and complexity == "high":
        return client_factory.get_client("anthropic-claude")
    elif complexity == "low":
        return client_factory.get_client("gpt-3.5-turbo")
    else:
        return client_factory.get_client("default")

# Usage
creative_client = get_client_for_task("creative", "high")
response = creative_client.generate("Write a poem about artificial intelligence.")

factual_client = get_client_for_task("factual", "low")
response = factual_client.generate("What is the capital of France?")
```

## Batch Processing

### Processing Multiple Prompts

```python
from src.llm.batch import BatchProcessor
from src.llm.client import LLMClient

# Initialize client
client = LLMClient.from_config("config/model.yaml")

# Initialize batch processor
batch_processor = BatchProcessor(client, max_concurrent=5)

# Prepare prompts
prompts = [
    "Summarize the benefits of exercise.",
    "List 5 healthy breakfast ideas.",
    "Explain how to start meditation.",
    "Provide tips for better sleep.",
    "Describe the benefits of drinking water."
]

# Process batch
results = batch_processor.process(prompts)

# Print results
for prompt, result in zip(prompts, results):
    print(f"Prompt: {prompt[:30]}...")
    print(f"Response: {result.text[:100]}...")
    print("-" * 50)
```

## Evaluation Examples

### Evaluating Response Quality

```python
from src.evaluation.metrics import RelevanceEvaluator, CoherenceEvaluator
from src.llm.client import LLMClient

# Initialize components
client = LLMClient.from_config("config/model.yaml")
relevance_evaluator = RelevanceEvaluator()
coherence_evaluator = CoherenceEvaluator()

# Sample prompt and response
prompt = "Explain how blockchain technology works."
response = client.generate(prompt)

# Evaluate response
relevance_score = relevance_evaluator.evaluate(prompt, response.text)
coherence_score = coherence_evaluator.evaluate(response.text)

print(f"Response: {response.text[:100]}...")
print(f"Relevance Score: {relevance_score}/10")
print(f"Coherence Score: {coherence_score}/10")
```

### A/B Testing Prompts

```python
from src.evaluation.ab_testing import ABTester
from src.llm.client import LLMClient
from src.prompt_engineering.templates import PromptTemplate

# Initialize components
client = LLMClient.from_config("config/model.yaml")
ab_tester = ABTester(client)

# Define prompt variants
template_a = PromptTemplate(
    template="Write a short advertisement for {product}.",
    input_variables=["product"]
)

template_b = PromptTemplate(
    template="Create a compelling advertisement for {product}. Include key benefits and a call to action.",
    input_variables=["product"]
)

# Test parameters
test_params = {
    "product": "eco-friendly water bottle",
    "num_samples": 10,
    "metrics": ["length", "engagement_score", "call_to_action_present"]
}

# Run A/B test
results = ab_tester.test(template_a, template_b, test_params)

# Print results
print("A/B Test Results:")
print(f"Template A average scores: {results['template_a_avg']}")
print(f"Template B average scores: {results['template_b_avg']}")
print(f"Winner: {results['winner']}")
```

These examples demonstrate the core functionality of our Generative AI framework. For more detailed examples and advanced use cases, refer to the example notebooks in the `notebooks/` directory.
