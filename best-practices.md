# Generative AI Best Practices

This document outlines best practices for developing, deploying, and maintaining Generative AI applications using our framework.

## Table of Contents

1. [Prompt Engineering](#prompt-engineering)
2. [Model Selection and Usage](#model-selection-and-usage)
3. [Data Management](#data-management)
4. [Performance Optimization](#performance-optimization)
5. [Error Handling](#error-handling)
6. [Security and Safety](#security-and-safety)
7. [Testing and Evaluation](#testing-and-evaluation)
8. [Deployment and Operations](#deployment-and-operations)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Ethical Considerations](#ethical-considerations)

## Prompt Engineering

### Effective Prompt Design

#### 1. Be Specific and Clear
- **DO**: Provide clear, specific instructions with examples
- **DON'T**: Use vague or ambiguous language

```python
# Good Example
prompt = """
Generate a product description for a wireless headphone with the following characteristics:
- Noise cancellation
- 30-hour battery life
- Bluetooth 5.0 connectivity
- Memory foam ear cushions

The description should be 3-4 sentences long and highlight comfort and sound quality.
"""

# Poor Example
prompt = "Write something about headphones."
```

#### 2. Use Structured Formats
- **DO**: Provide clear formatting instructions and examples
- **DON'T**: Assume the model will infer your desired format

```python
# Good Example
prompt = """
Create a recipe in the following format:

Name: [Recipe Name]
Prep Time: [Time in minutes]
Ingredients:
- [Ingredient 1]
- [Ingredient 2]
...
Instructions:
1. [Step 1]
2. [Step 2]
...

Create a recipe for a vegetarian pasta dish.
"""
```

#### 3. Leverage System Messages
- **DO**: Use system messages to set the tone and context
- **DON'T**: Put all instructions in the user message

```python
# Good Example
system_message = "You are a professional copywriter specializing in technical products."
user_message = "Write a product description for our new AI-powered smart speaker."

# Using the ConversationManager
conversation = ConversationManager(system_message=system_message)
conversation.add_user_message(user_message)
```

### Prompt Versioning and Management

#### 1. Version Your Prompts
- **DO**: Maintain version control for prompts
- **DON'T**: Modify prompts without tracking changes

```python
# Example: Versioned prompt template
@prompt_version("1.2.0")
class ProductDescriptionTemplate(BasePromptTemplate):
    """Generate product descriptions based on attributes."""
    # Template implementation
```

#### 2. Document Prompt Intent and Behavior
- **DO**: Include clear documentation for each prompt template
- **DON'T**: Create prompts without explaining their purpose and expected output

```python
class MarketingEmailTemplate(BasePromptTemplate):
    """
    Generates marketing emails for product announcements.
    
    Expected inputs:
    - product_name: Name of the product being announced
    - key_features: List of key product features (3-5 items)
    - target_audience: Description of the target customer
    - call_to_action: The specific action you want readers to take
    
    Output format:
    A marketing email with subject line, greeting, 2-3 paragraphs of body text,
    and a call to action. Approximately 200-300 words.
    """
    # Template implementation
```

## Model Selection and Usage

### Choosing the Right Model

#### 1. Match Model to Task Complexity
- **DO**: Use simpler, faster models for straightforward tasks
- **DON'T**: Default to the most powerful model for every task

| Task Type | Recommended Model |
|-----------|------------------|
| Simple text completion | GPT-3.5 Turbo |
| Creative writing | Claude 3 Opus or GPT-4 |
| Code generation | CodeLlama or GPT-4 |
| Factual Q&A | Claude 3 Sonnet or GPT-4 |
| Summarization | Mistral or GPT-3.5 Turbo |

#### 2. Consider Cost vs. Performance
- **DO**: Benchmark different models for your specific use case
- **DON'T**: Assume the most expensive model is always best

```python
# Example: Model benchmarking utility
def benchmark_models(prompt, models, metrics, n_runs=5):
    """Benchmark multiple models against the same prompt."""
    results = {}
    
    for model_name in models:
        client = LLMClientFactory().get_client(model_name)
        model_results = []
        
        for _ in range(n_runs):
            response = client.generate(prompt)
            scores = {metric: evaluators[metric].evaluate(prompt, response.text) 
                     for metric in metrics}
            model_results.append(scores)
        
        results[model_name] = {
            metric: sum(run[metric] for run in model_results) / n_runs
            for metric in metrics
        }
    
    return results
```

### Optimizing Model Parameters

#### 1. Temperature and Sampling
- **DO**: Use lower temperature (0.1-0.4) for factual or structured outputs
- **DO**: Use higher temperature (0.7-1.0) for creative tasks
- **DON'T**: Use temperature 0 unless you need deterministic outputs

```python
# Factual query configuration
factual_config = {
    "model": "gpt-4",
    "temperature": 0.2,
    "max_tokens": 500
}

# Creative writing configuration
creative_config = {
    "model": "gpt-4",
    "temperature": 0.8,
    "max_tokens": 1000
}
```

#### 2. Context Window Management
- **DO**: Track token usage and optimize prompt length
- **DON'T**: Exceed model context limits or waste tokens on irrelevant information

```python
# Example: Context window management
def optimize_context(query, documents, max_tokens, tokenizer):
    """Optimize context selection to fit within token limits."""
    query_tokens = len(tokenizer.encode(query))
    available_tokens = max_tokens - query_tokens - 100  # Buffer for response
    
    selected_docs = []
    total_tokens = 0
    
    for doc in sorted(documents, key=lambda d: d["relevance"], reverse=True):
        doc_tokens = len(tokenizer.encode(doc["text"]))
        if total_tokens + doc_tokens <= available_tokens:
            selected_docs.append(doc["text"])
            total_tokens += doc_tokens
        else:
            break
    
    return "\n\n".join(selected_docs)
```

## Data Management

### Vector Database Best Practices

#### 1. Chunking Strategy
- **DO**: Use semantic chunking based on content structure
- **DON'T**: Use fixed-size chunks that might split related information

```python
# Example: Semantic chunking
def semantic_chunking(document):
    """Chunk document based on semantic boundaries."""
    # Split by sections/paragraphs first
    sections = split_by_section_headers(document)
    chunks = []
    
    for section in sections:
        # If section is too large, split further by paragraphs
        if len(tokenizer.encode(section)) > MAX_CHUNK_TOKENS:
            paragraphs = split_by_paragraphs(section)
            chunks.extend(paragraphs)
        else:
            chunks.append(section)
    
    # Further processing for very large paragraphs if needed
    
    return chunks
```

#### 2. Metadata and Filtering
- **DO**: Store rich metadata with embeddings for filtering
- **DON'T**: Rely solely on vector similarity for retrieval

```python
# Example: Adding documents with metadata
def add_document_to_vector_db(document, vector_db):
    """Add document with metadata to vector database."""
    chunks = semantic_chunking(document)
    
    for chunk in chunks:
        # Extract metadata
        metadata = {
            "source": document.source,
            "author": document.author,
            "created_at": document.created_at,
            "category": document.category,
            "section": extract_section(chunk),
            "chunk_id": generate_chunk_id(chunk)
        }
        
        # Generate embedding
        embedding = embedding_generator.generate(chunk)
        
        # Store in vector database
        vector_db.add(chunk, embedding, metadata)
```

### Prompt Template Management

#### 1. Centralized Template Storage
- **DO**: Store templates in a centralized repository
- **DON'T**: Duplicate templates across different parts of the application

```python
# Example: Template registry
class PromptTemplateRegistry:
    """Registry for prompt templates."""
    
    def __init__(self):
        self.templates = {}
    
    def register(self, name, template, version="latest"):
        """Register a template with version."""
        if name not in self.templates:
            self.templates[name] = {}
        self.templates[name][version] = template
    
    def get(self, name, version="latest"):
        """Get a template by name and version."""
        if name not in self.templates:
            raise ValueError(f"Template {name} not found")
        
        if version not in self.templates[name]:
            if version == "latest":
                # Get the latest version
                version = max(self.templates[name].keys())
            else:
                raise ValueError(f"Version {version} of template {name} not found")
        
        return self.templates[name][version]
```

#### 2. Template Validation
- **DO**: Validate templates before use
- **DON'T**: Use templates with missing or invalid parameters

```python
# Example: Template validation
def validate_template(template, required_variables):
    """Validate that a template contains all required variables."""
    for variable in required_variables:
        if f"{{{variable}}}" not in template.template:
            raise ValueError(f"Template missing required variable: {variable}")
```

## Performance Optimization

### Rate Limiting and Batching

#### 1. Implement Adaptive Rate Limiting
- **DO**: Use adaptive rate limiting based on provider responses
- **DON'T**: Use fixed delays that don't adjust to actual limits

```python
# Example: Adaptive rate limiter
class AdaptiveRateLimiter:
    """Rate limiter that adapts to provider responses."""
    
    def __init__(self, initial_rpm=60):
        self.rpm = initial_rpm
        self.last_request_time = 0
        self.backoff_factor = 1.0
    
    def wait_if_needed(self):
        """Wait if necessary to comply with rate limits."""
        current_time = time.time()
        min_interval = 60 / self.rpm
        elapsed = current_time - self.last_request_time
        
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        
        self.last_request_time = time.time()
    
    def handle_success(self):
        """Handle successful request."""
        # Gradually increase rate if we've been successful
        if self.backoff_factor > 1.0:
            self.backoff_factor = max(1.0, self.backoff_factor * 0.95)
            self.rpm = min(initial_rpm, self.rpm / self.backoff_factor)
    
    def handle_rate_limit_error(self):
        """Handle rate limit error."""
        # Reduce rate limit and apply backoff
        self.backoff_factor *= 1.5
        self.rpm = self.rpm / self.backoff_factor
```

#### 2. Optimize Batch Processing
- **DO**: Group similar requests for batch processing
- **DON'T**: Process each request individually when batching is available

```python
# Example: Smart batching
class SmartBatcher:
    """Intelligently batch requests based on similarity and priority."""
    
    def __init__(self, client, max_batch_size=20, similarity_threshold=0.8):
        self.client = client
        self.max_batch_size = max_batch_size
        self.similarity_threshold = similarity_threshold
        self
