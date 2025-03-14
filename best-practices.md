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
        self.embedding_generator = EmbeddingGenerator()
        self.queue = []
    
    def add_to_queue(self, prompt, callback, priority=1):
        """Add a prompt to the processing queue."""
        embedding = self.embedding_generator.generate(prompt)
        self.queue.append({
            "prompt": prompt,
            "embedding": embedding,
            "callback": callback,
            "priority": priority,
            "added_at": time.time()
        })
    
    def process_queue(self):
        """Process the queue in optimized batches."""
        if not self.queue:
            return
        
        # Sort by priority and age
        self.queue.sort(key=lambda x: (-x["priority"], x["added_at"]))
        
        # Create batches of similar prompts
        batches = []
        remaining = self.queue.copy()
        
        while remaining:
            batch = [remaining.pop(0)]
            batch_embedding = batch[0]["embedding"]
            
            # Find similar prompts for the batch
            i = 0
            while i < len(remaining) and len(batch) < self.max_batch_size:
                similarity = cosine_similarity(batch_embedding, remaining[i]["embedding"])
                if similarity > self.similarity_threshold:
                    batch.append(remaining.pop(i))
                else:
                    i += 1
            
            batches.append(batch)
        
        # Process each batch
        for batch in batches:
            prompts = [item["prompt"] for item in batch]
            responses = self.client.batch_generate(prompts)
            
            # Call callbacks with responses
            for item, response in zip(batch, responses):
                item["callback"](response)
        
        self.queue = []
```

### Caching Strategies

#### 1. Implement Multi-level Caching
- **DO**: Use different caching strategies for different types of requests
- **DON'T**: Apply the same caching policy to all responses

```python
# Example: Multi-level cache implementation
class MultiLevelCache:
    """Multi-level cache for LLM responses."""
    
    def __init__(self):
        # Fast, in-memory cache for exact matches
        self.exact_cache = {}
        
        # Vector cache for semantic similarity
        self.vector_cache = VectorDatabase()
        
        # Persistent cache for long-term storage
        self.persistent_cache = SqliteCache("cache.db")
        
        self.embedding_generator = EmbeddingGenerator()
    
    def get(self, prompt, similarity_threshold=0.95):
        """Get a cached response for a prompt."""
        # Check exact cache first
        if prompt in self.exact_cache:
            return self.exact_cache[prompt]
        
        # Check vector cache for similar prompts
        prompt_embedding = self.embedding_generator.generate(prompt)
        similar_entries = self.vector_cache.search(prompt_embedding, top_k=5)
        
        for entry in similar_entries:
            if entry["similarity"] > similarity_threshold:
                return entry["response"]
        
        # Check persistent cache
        return self.persistent_cache.get(prompt)
    
    def set(self, prompt, response):
        """Cache a response."""
        # Add to exact cache
        self.exact_cache[prompt] = response
        
        # Add to vector cache
        prompt_embedding = self.embedding_generator.generate(prompt)
        self.vector_cache.add(prompt, prompt_embedding, {"response": response})
        
        # Add to persistent cache
        self.persistent_cache.set(prompt, response)
```

#### 2. Cache Invalidation Strategy
- **DO**: Implement time-based and version-based cache invalidation
- **DON'T**: Cache responses indefinitely without a clear invalidation strategy

```python
# Example: Cache with invalidation
class CacheWithInvalidation:
    """Cache with invalidation strategies."""
    
    def __init__(self, default_ttl=3600):
        self.cache = {}
        self.default_ttl = default_ttl
        self.version_keys = {}
    
    def get(self, key, version=None):
        """Get a value from the cache."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        
        # Check if entry has expired
        if entry["expires_at"] < time.time():
            del self.cache[key]
            return None
        
        # Check if version matches
        if version and entry["version"] != version:
            return None
        
        return entry["value"]
    
    def set(self, key, value, ttl=None, version=None):
        """Set a value in the cache."""
        expires_at = time.time() + (ttl or self.default_ttl)
        
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at,
            "version": version
        }
        
        # Track keys by version for bulk invalidation
        if version:
            if version not in self.version_keys:
                self.version_keys[version] = set()
            self.version_keys[version].add(key)
    
    def invalidate_version(self, version):
        """Invalidate all entries for a specific version."""
        if version not in self.version_keys:
            return
        
        for key in self.version_keys[version]:
            if key in self.cache and self.cache[key]["version"] == version:
                del self.cache[key]
        
        del self.version_keys[version]
```

## Error Handling

### Robust Error Management

#### 1. Implement Comprehensive Error Handling
- **DO**: Handle all potential error types with specific strategies
- **DON'T**: Use generic try/except blocks without specific error handling

```python
# Example: Comprehensive error handling
def generate_with_error_handling(client, prompt, max_retries=3):
    """Generate a response with comprehensive error handling."""
    retries = 0
    backoff = 1
    
    while retries <= max_retries:
        try:
            return client.generate(prompt)
        
        except RateLimitError as e:
            # Handle rate limiting
            logger.warning(f"Rate limit exceeded: {e}")
            time.sleep(backoff)
            backoff *= 2
            retries += 1
        
        except ContextLengthExceededError as e:
            # Handle context length issues
            logger.warning(f"Context length exceeded: {e}")
            shortened_prompt = truncate_prompt(prompt)
            return client.generate(shortened_prompt)
        
        except ContentFilterError as e:
            # Handle content filtering
            logger.warning(f"Content filter triggered: {e}")
            sanitized_prompt = sanitize_prompt(prompt)
            return client.generate(sanitized_prompt)
        
        except AuthenticationError as e:
            # Handle authentication issues
            logger.error(f"Authentication error: {e}")
            refresh_credentials()
            retries += 1
        
        except ServiceUnavailableError as e:
            # Handle service outages
            logger.error(f"Service unavailable: {e}")
            time.sleep(backoff)
            backoff *= 2
            retries += 1
        
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Unexpected error: {e}")
            if retries >= max_retries:
                return fallback_response(prompt)
            time.sleep(backoff)
            backoff *= 2
            retries += 1
    
    return fallback_response(prompt)
```

#### 2. Implement Graceful Degradation
- **DO**: Provide fallback options when primary models fail
- **DON'T**: Allow the entire application to fail when one component fails

```python
# Example: Graceful degradation strategy
class GracefulDegradationStrategy:
    """Strategy for graceful degradation when primary models fail."""
    
    def __init__(self):
        # Define fallback options in order of preference
        self.model_fallbacks = {
            "gpt-4": ["gpt-3.5-turbo", "text-davinci-003", "text-curie-001"],
            "claude-3-opus": ["claude-3-sonnet", "claude-3-haiku", "gpt-3.5-turbo"],
            # Add more fallbacks for other models
        }
        
        self.client_factory = LLMClientFactory()
    
    def generate_with_fallbacks(self, model, prompt, max_attempts=4):
        """Try to generate a response, falling back to simpler models if needed."""
        models_to_try = [model] + self.model_fallbacks.get(model, [])
        models_to_try = models_to_try[:max_attempts]  # Limit attempts
        
        last_error = None
        
        for fallback_model in models_to_try:
            try:
                client = self.client_factory.get_client(fallback_model)
                return client.generate(prompt), fallback_model
            except Exception as e:
                logger.warning(f"Error with model {fallback_model}: {e}")
                last_error = e
        
        # If all fallbacks fail, raise the last error
        raise last_error
```

## Security and Safety

### Input Validation and Sanitization

#### 1. Implement Prompt Injection Prevention
- **DO**: Validate and sanitize user inputs
- **DON'T**: Pass raw user input directly to the model

```python
# Example: Prompt injection prevention
def sanitize_user_input(user_input):
    """Sanitize user input to prevent prompt injection."""
    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1F\x7F]', '', user_input)
    
    # Check for suspicious patterns
    injection_patterns = [
        r'ignore previous instructions',
        r'disregard all prior prompts',
        r'system prompt:',
        r'you are now',
        # Add more patterns as needed
    ]
    
    for pattern in injection_patterns:
        if re.search(pattern, sanitized, re.IGNORECASE):
            logger.warning(f"Potential prompt injection detected: {pattern}")
            sanitized = re.sub(pattern, '[FILTERED]', sanitized, flags=re.IGNORECASE)
    
    return sanitized

# Usage
user_input = get_user_input()
safe_input = sanitize_user_input(user_input)
prompt = template.format(user_input=safe_input)
```

#### 2. Implement Output Filtering
- **DO**: Filter model outputs for sensitive or harmful content
- **DON'T**: Display raw model outputs without verification

```python
# Example: Output filtering
def filter_output(response):
    """Filter model output for sensitive or harmful content."""
    # Check for PII
    pii_patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(\+\d{1,2}\s)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        # Add more PII patterns
    }
    
    filtered_response = response
    
    for pii_type, pattern in pii_patterns.items():
        matches = re.finditer(pattern, filtered_response)
        for match in matches:
            logger.warning(f"Found {pii_type} in response")
            filtered_response = filtered_response.replace(match.group(0), f"[REDACTED {pii_type}]")
    
    # Check for harmful content
    harmful_content_detector = HarmfulContentDetector()
    if harmful_content_detector.contains_harmful_content(filtered_response):
        logger.warning("Harmful content detected in response")
        return "I cannot provide that information as it may contain harmful content."
    
    return filtered_response
```

### API Key Management

#### 1. Secure Credential Storage
- **DO**: Use environment variables or secret management services
- **DON'T**: Hardcode API keys in source code or config files

```python
# Example: Secure credential management
class SecureCredentialManager:
    """Manage API credentials securely."""
    
    def __init__(self):
        # Try to load from environment variables first
        self.api_keys = {}
        
        # Connect to secret manager if available
        try:
            self.secret_manager = connect_to_secret_manager()
            self.use_secret_manager = True
        except Exception:
            self.use_secret_manager = False
    
    def get_api_key(self, provider):
        """Get API key for a provider."""
        # Check if we already have the key
        if provider in self.api_keys:
            return self.api_keys[provider]
        
        # Try to get from environment
        env_var = f"{provider.upper()}_API_KEY"
        api_key = os.environ.get(env_var)
        
        if api_key:
            self.api_keys[provider] = api_key
            return api_key
        
        # Try to get from secret manager
        if self.use_secret_manager:
            secret_name = f"{provider}-api-key"
            api_key = self.secret_manager.get_secret(secret_name)
            
            if api_key:
                self.api_keys[provider] = api_key
                return api_key
        
        raise ValueError(f"No API key found for {provider}")
```

## Testing and Evaluation

### Comprehensive Testing Strategy

#### 1. Implement Prompt Regression Testing
- **DO**: Create a suite of test prompts with expected outputs
- **DON'T**: Rely solely on manual testing for prompt quality

```python
# Example: Prompt regression testing
class PromptRegressionTester:
    """Test prompts against expected outputs."""
    
    def __init__(self, client):
        self.client = client
        self.test_cases = []
    
    def add_test_case(self, prompt, expected_outputs, similarity_threshold=0.8):
        """Add a test case."""
        self.test_cases.append({
            "prompt": prompt,
            "expected_outputs": expected_outputs if isinstance(expected_outputs, list) else [expected_outputs],
            "similarity_threshold": similarity_threshold
        })
    
    def load_test_cases_from_file(self, filepath):
        """Load test cases from a YAML file."""
        with open(filepath, 'r') as f:
            test_cases = yaml.safe_load(f)
        
        for test_case in test_cases:
            self.add_test_case(
                test_case["prompt"],
                test_case["expected_outputs"],
                test_case.get("similarity_threshold", 0.8)
            )
    
    def run_tests(self):
        """Run all test cases."""
        results = []
        
        for test_case in self.test_cases:
            response = self.client.generate(test_case["prompt"])
            
            # Check if response matches any expected output
            best_similarity = 0
            for expected in test_case["expected_outputs"]:
                similarity = semantic_similarity(response.text, expected)
                best_similarity = max(best_similarity, similarity)
            
            passed = best_similarity >= test_case["similarity_threshold"]
            
            results.append({
                "prompt": test_case["prompt"],
                "response": response.text,
                "best_similarity": best_similarity,
                "passed": passed
            })
        
        return results
```

#### 2. Implement A/B Testing Framework
- **DO**: Systematically compare different prompt versions
- **DON'T**: Make subjective judgments without quantitative metrics

```python
# Example: A/B testing framework
class PromptABTester:
    """A/B test different prompt versions."""
    
    def __init__(self, client, evaluators):
        self.client = client
        self.evaluators = evaluators
    
    def test(self, prompt_a, prompt_b, n_samples=100):
        """Compare two prompts across multiple samples."""
        results_a = []
        results_b = []
        
        for _ in range(n_samples):
            # Generate responses
            response_a = self.client.generate(prompt_a)
            response_b = self.client.generate(prompt_b)
            
            # Evaluate responses
            scores_a = {name: evaluator.evaluate(response_a.text) 
                       for name, evaluator in self.evaluators.items()}
            scores_b = {name: evaluator.evaluate(response_b.text)
                       for name, evaluator in self.evaluators.items()}
            
            results_a.append(scores_a)
            results_b.append(scores_b)
        
        # Aggregate results
        avg_a = {name: sum(result[name] for result in results_a) / n_samples
                for name in self.evaluators.keys()}
        avg_b = {name: sum(result[name] for result in results_b) / n_samples
                for name in self.evaluators.keys()}
        
        # Determine winner for each metric
        winners = {}
        for name in self.evaluators.keys():
            if avg_a[name] > avg_b[name]:
                winners[name] = "A"
            elif avg_b[name] > avg_a[name]:
                winners[name] = "B"
            else:
                winners[name] = "Tie"
        
        return {
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
            "avg_scores_a": avg_a,
            "avg_scores_b": avg_b,
            "winners": winners
        }
```

## Deployment and Operations

### Deployment Best Practices

#### 1. Implement Blue-Green Deployments
- **DO**: Use blue-green deployment for zero-downtime updates
- **DON'T**: Deploy directly to production without testing in a staging environment

```python
# Example: Blue-green deployment configuration
"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service-blue
spec:
  replicas: 3
  selector:
    matchLabels:
      app: llm-service
      deployment: blue
  template:
    metadata:
      labels:
        app: llm-service
        deployment: blue
    spec:
      containers:
      - name: llm-service
        image: llm-service:1.0.0
        env:
        - name: MODEL_VERSION
          value: "2023-05-01"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-service-green
spec:
  replicas: 0  # Start with 0 replicas
  selector:
    matchLabels:
      app: llm-service
      deployment: green
  template:
    metadata:
      labels:
        app: llm-service
        deployment: green
    spec:
      containers:
      - name: llm-service
        image: llm-service:1.1.0  # New version
        env:
        - name: MODEL_VERSION
          value: "2023-06-15"  # New model version
---
apiVersion: v1
kind: Service
metadata:
  name: llm-service
spec:
  selector:
    app: llm-service
    deployment: blue  # Initially route to blue
  ports:
  - port: 80
    targetPort: 8080
"""
```

#### 2. Implement Canary Deployments
- **DO**: Gradually roll out changes to a subset of users
- **DON'T**: Release new versions to all users simultaneously

```python
# Example: Canary deployment script
def canary_deployment(new_version, initial_percent=5, increment=10, interval_minutes=15, max_percent=100):
    """Implement a canary deployment strategy."""
    # Deploy the new version with initial traffic percentage
    deploy_version(new_version, traffic_percent=initial_percent)
    
    current_percent = initial_percent
    while current_percent < max_percent:
        # Wait for the specified interval
        time.sleep(interval_minutes * 60)
        
        # Check error rates and performance metrics
        error_rate = get_error_rate(new_version)
        latency = get_average_latency(new_version)
        
        # If metrics are acceptable, increase the traffic percentage
        if error_rate < ERROR_THRESHOLD and latency < LATENCY_THRESHOLD:
            current_percent = min(current_percent + increment, max_percent)
            update_traffic_split(new_version, current_percent)
            logger.info(f"Increased traffic to {new_version} to {current_percent}%")
        else:
            # If metrics are poor, rollback
            logger.error(f"Metrics exceeded thresholds: error_rate={error_rate}, latency={latency}")
            rollback(new_version)
            return False
    
    # Deployment complete
    logger.info(f"Canary deployment of {new_version} complete")
    return True
```

### Scaling Strategies

#### 1. Implement Horizontal Scaling
- **DO**: Design services to scale horizontally with load
- **DON'T**: Rely on vertical scaling for high-traffic applications

```python
# Example: Horizontal scaling configuration
"""
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: llm-service-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: llm-service
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: 100
"""
```

## Monitoring and Maintenance

### Comprehensive Monitoring

#### 1. Implement Multi-dimensional Monitoring
- **DO**: Monitor technical, business, and quality metrics
- **DON'T**: Focus solely on technical metrics like latency and error rates

```python
# Example: Multi-dimensional monitoring setup
class LLMMonitoring:
    """Comprehensive monitoring for LLM applications."""
    
    def __init__(self):
        # Technical metrics
        self.technical_metrics = {
            "latency": Histogram("response_latency_seconds", "Response latency in seconds"),
            "error_rate": Counter("error_count", "Number of errors"),
            "token_usage": Counter("token_usage", "Number of tokens used", ["model", "prompt_type"]),
            "request_count": Counter("request_count", "Number of requests", ["model", "endpoint"])
        }
        
        # Business metrics
        self.business_metrics = {
            "conversion_rate": Gauge("conversion_rate", "User conversion rate"),
            "user_engagement": Histogram("user_engagement_seconds", "User engagement time"),
            "feature_usage": Counter("feature_usage", "Feature usage count", ["feature"])
        }
        
        # Quality metrics
        self.quality_metrics = {
            "relevance_score": Histogram("relevance_score", "Response relevance score"),
            "coherence_score": Histogram("coherence_score", "Response coherence score"),
            "helpfulness_score": Histogram("helpfulness_score", "Response helpfulness score"),
            "user_satisfaction": Histogram("user_satisfaction", "User satisfaction rating")
        }
    
    def record_request(self, model, endpoint, start_time, tokens, prompt_type):
        """Record metrics for a request."""
        # Technical metrics
        latency = time.time() - start_time
        self.technical_metrics["latency"].observe(latency)
        self.technical_metrics["request_count"].labels(model=model, endpoint=endpoint).inc()
        self.technical_metrics["token_usage"].labels(model=model, prompt_type=prompt_type).inc(tokens)
    
    def record_error(self, error_type):
        """Record an error."""
        self.technical_metrics["error_rate"].inc()
        # Additional error-specific metrics could be added here
    
    def record_quality_metrics(self, response, prompt, user_feedback=None):
        """Record quality metrics for a response."""
        # Calculate quality scores
        relevance = calculate_relevance(prompt, response)
        coherence = calculate_coherence(response)
        helpfulness = calculate_helpfulness(response)
        
        # Record metrics
        self.quality_metrics["relevance_score"].observe(relevance)
        self.quality_metrics["coherence_score"].observe(coherence)
        self.quality_metrics["helpfulness_score"].observe(helpfulness)
        
        # Record user feedback if available
        if user_feedback:
            self.quality_metrics["user_satisfaction"].observe(user_feedback)
    
    def record_business_metrics(self, user_id, session_data):
        """Record business metrics."""
        if session_data.get("converted"):
            self.business_metrics["conversion_rate"].inc()
        
        if session_data.get("engagement_time"):
            self.business_metrics["user_engagement"].observe(session_data["engagement_time"])
        
        if session_data.get("features_used"):
            for feature in session_data["features_used"]:
                self.business_metrics["feature_usage"].labels(feature=feature).inc()
```

#### 2. Implement Alerting and Dashboards
- **DO**: Set up alerts for critical issues and comprehensive dashboards
- **DON'T**: Rely on manual monitoring or generic dashboards

```python
# Example: Alerting configuration
"""
groups:
- name: LLM Service Alerts
  rules:
  - alert: HighErrorRate
    expr: sum(rate(error_count[5m])) / sum(rate(request_count[5m])) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 5% for 5 minutes"
  
  - alert: HighLatency
    expr: histogram_quantile(0.95, sum(rate(response_latency_seconds_bucket[5m])) by (le)) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is above 2 seconds for 5 minutes"
  
  - alert: TokenQuotaNearLimit
    expr: sum(increase(token_usage[24h])) > 0.8 * TOKEN_QUOTA
    labels:
      severity: warning
    annotations:
      summary: "Token quota nearly exhausted"
      description: "Token usage is above 80% of daily quota"
  
  - alert: LowQualityResponses
    expr: avg(relevance_score) < 0.7
    for: 30m
    labels:
      severity: warning
    annotations:
      summary: "Low quality responses detected"
      description: "Average relevance score is below 0.7 for 30 minutes"
"""
```

## Ethical Considerations

### Responsible AI Practices

#### 1. Implement Fairness Monitoring
- **DO**: Monitor and mitigate bias in model outputs
- **DON'T**: Assume models are inherently fair or unbiased

```python
# Example: Bias monitoring
class BiasMonitor:
    """Monitor and detect bias in model outputs."""
    
    def __init__(self):
        self.sensitive_categories = [
            "gender", "race", "ethnicity", "religion", "age", 
            "disability", "sexual_orientation", "nationality"
        ]
        
        self.bias_metrics = {
            category: {
                "representation": Counter(f"representation_{category}", f"Representation of {category}"),
                "sentiment": Histogram(f"sentiment_{category}", f"Sentiment distribution for {category}")
            }
            for category in self.sensitive_categories
        }
        
        self.bias_detector = load_bias_detection_model()
    
    def analyze_response(self, response):
        """Analyze a response for potential bias."""
        # Detect mentions of sensitive categories
        category_mentions = self.bias_detector.detect_categories(response)
        
        # Record metrics for each detected category
        for category, mentions in category_mentions.items():
            if category in self.sensitive_categories and mentions:
                # Record representation
                self.bias_metrics[category]["representation"].inc(len(mentions))
                
                # Analyze sentiment for each mention
                for mention in mentions:
                    sentiment = self.bias_detector.analyze_sentiment(mention["context"])
                    self.bias_metrics[category]["sentiment"].observe(sentiment)
        
        # Detect potentially problematic content
        issues = self.bias_detector.detect_issues(response)
        
        return {
            "category_mentions": category_mentions,
            "issues": issues
        }
    
    def generate_bias_report(self, timeframe="day"):
        """Generate a report on bias metrics."""
        report = {}
        
        for category in self.sensitive_categories:
            representation = get_metric_value(self.bias_metrics[category]["representation"], timeframe)
            sentiment_distribution = get_histogram_distribution(self.bias_metrics[category]["sentiment"], timeframe)
            sentiment_avg = calculate_average(sentiment_distribution)
            
            report[category] = {
                "representation": representation,
                "sentiment_avg": sentiment_avg,
                "sentiment_distribution": sentiment_distribution
            }
        
        return report
```

#### 2. Implement Transparency Measures
- **DO**: Clearly disclose AI-generated content and model limitations
- **DON'T**: Mislead users about AI capabilities or involvement

```python
# Example: Transparency implementation
class TransparencyManager:
    """Manage transparency for AI-generated content."""
    
    def __init__(self):
        self.model_cards = load_model_cards()
    
    def get_disclosure_notice(self, model, context="general"):
        """Get an appropriate disclosure notice."""
        base_notice = "This content was generated by an AI system."
        
        if model in self.model_cards:
            model_info = self.model_cards[model]
            
            # Add model-specific information
            base_notice += f" Model: {model_info['name']} ({model_info['version']})"
            
            # Add context-specific limitations
            if context in model_info["limitations"]:
                base_notice += f" Note: {model_info['limitations'][context]}"
        
        return base_notice
    
    def add_disclosure_to_response(self, response, model, context="general"):
        """Add a disclosure notice to a response."""
        notice = self.get_disclosure_notice(model, context)
        
        # Add the notice in an appropriate way based on the response format
        if isinstance(response, dict) and "text" in response:
            response["disclosure"] = notice
        elif isinstance(response, str):
            response = f"{response}\n\n{notice}"
        
        return response
    
 def generate_model_card_page(self, model):
        """Generate a model card page for transparency."""
        if model not in self.model_cards:
            return f"Model card for {model} not available."
        
        model_info = self.model_cards[model]
        
        return f"""
        # Model Card: {model_info['name']}
        
        ## Basic Information
        - **Model Version:** {model_info['version']}
        - **Release Date:** {model_info['release_date']}
        - **Type:** {model_info['type']}
        - **Developer:** {model_info['developer']}
        
        ## Intended Use
        {model_info['intended_use']}
        
        ## Training Data
        {model_info['training_data']}
        
        ## Performance and Limitations
        {model_info['performance']}
        
        ### Known Limitations
        {model_info['limitations']['general']}
        
        ## Ethical Considerations
        {model_info['ethical_considerations']}
        
        ## Contact Information
        For questions or concerns about this model, please contact:
        {model_info['contact']}
        """
```

#### 3. Implement Content Moderation
- **DO**: Use multi-layered content moderation for both inputs and outputs
- **DON'T**: Rely solely on model-based filtering without human oversight

```python
# Example: Content moderation system
class ContentModerationSystem:
    """Multi-layered content moderation system."""
    
    def __init__(self):
        # Load moderation components
        self.keyword_filter = KeywordFilter("data/moderation/keywords.json")
        self.ml_classifier = ToxicityClassifier()
        self.pattern_matcher = RegexPatternMatcher("data/moderation/patterns.json")
        
        # Set up human review queue
        self.review_queue = HumanReviewQueue()
        
        # Configure moderation thresholds
        self.thresholds = {
            "auto_reject": 0.9,  # Automatically reject if score exceeds this
            "auto_accept": 0.1,  # Automatically accept if score below this
            "human_review": 0.5  # Send for human review if score exceeds this
        }
    
    def moderate_input(self, user_input, context=None):
        """Moderate user input."""
        # Apply keyword filtering
        keyword_result = self.keyword_filter.check(user_input)
        
        # Apply ML classification
        ml_result = self.ml_classifier.classify(user_input)
        
        # Apply pattern matching
        pattern_result = self.pattern_matcher.check(user_input)
        
        # Combine scores (implementation depends on specific approach)
        combined_score = self._combine_scores([
            keyword_result.score,
            ml_result.score,
            pattern_result.score
        ])
        
        # Determine action based on score
        if combined_score >= self.thresholds["auto_reject"]:
            return {
                "action": "reject",
                "score": combined_score,
                "reason": self._get_rejection_reason([keyword_result, ml_result, pattern_result])
            }
        elif combined_score >= self.thresholds["human_review"]:
            # Queue for human review
            review_id = self.review_queue.add(user_input, combined_score, context)
            return {
                "action": "review",
                "score": combined_score,
                "review_id": review_id
            }
        else:
            return {
                "action": "accept",
                "score": combined_score
            }
    
    def moderate_output(self, model_output, prompt=None):
        """Moderate model-generated output."""
        # Similar process as input moderation
        keyword_result = self.keyword_filter.check(model_output)
        ml_result = self.ml_classifier.classify(model_output)
        pattern_result = self.pattern_matcher.check(model_output)
        
        combined_score = self._combine_scores([
            keyword_result.score,
            ml_result.score,
            pattern_result.score
        ])
        
        if combined_score >= self.thresholds["auto_reject"]:
            return {
                "action": "reject",
                "score": combined_score,
                "reason": self._get_rejection_reason([keyword_result, ml_result, pattern_result])
            }
        elif combined_score >= self.thresholds["human_review"]:
            # For outputs, we might want to be more cautious
            if prompt:
                review_id = self.review_queue.add(model_output, combined_score, {"prompt": prompt})
                return {
                    "action": "review",
                    "score": combined_score,
                    "review_id": review_id
                }
            else:
                # Without context, reject borderline cases
                return {
                    "action": "reject",
                    "score": combined_score,
                    "reason": "Potentially problematic content requiring review"
                }
        else:
            return {
                "action": "accept",
                "score": combined_score
            }
    
    def _combine_scores(self, scores):
        """Combine multiple moderation scores."""
        # Could use max, weighted average, or more complex combination
        return max(scores)
    
    def _get_rejection_reason(self, results):
        """Get a user-friendly rejection reason."""
        reasons = []
        
        for result in results:
            if result.score > self.thresholds["human_review"]:
                reasons.append(result.reason)
        
        if reasons:
            return "This content was flagged for: " + ", ".join(reasons)
        else:
            return "This content was flagged by our moderation system."
```

### Data Privacy and Governance

#### 1. Implement Data Minimization
- **DO**: Collect and process only necessary data
- **DON'T**: Store sensitive data unnecessarily or longer than needed

```python
# Example: Data minimization implementation
class DataMinimizer:
    """Implement data minimization practices."""
    
    def __init__(self, pii_detector):
        self.pii_detector = pii_detector
        self.retention_policies = load_retention_policies()
    
    def minimize_input(self, user_input, purpose):
        """Minimize user input based on purpose."""
        # Detect PII in the input
        pii_detected = self.pii_detector.detect(user_input)
        
        # Check what PII is necessary for the stated purpose
        necessary_pii = self.retention_policies.get_necessary_pii(purpose)
        
        # Redact unnecessary PII
        minimized_input = user_input
        for pii_type, instances in pii_detected.items():
            if pii_type not in necessary_pii:
                for instance in instances:
                    minimized_input = minimized_input.replace(
                        instance["text"],
                        f"[REDACTED {pii_type}]"
                    )
        
        return {
            "minimized_input": minimized_input,
            "redacted_pii_types": [pii for pii in pii_detected if pii not in necessary_pii]
        }
    
    def apply_retention_policy(self, data, purpose):
        """Apply retention policy to stored data."""
        retention_period = self.retention_policies.get_retention_period(purpose)
        
        # Add expiration metadata
        expiration_date = datetime.now() + timedelta(days=retention_period)
        
        return {
            "data": data,
            "purpose": purpose,
            "expiration_date": expiration_date.isoformat()
        }
    
    def schedule_deletion(self, data_id, expiration_date):
        """Schedule data for deletion after retention period."""
        deletion_scheduler.schedule(
            task="delete_data",
            params={"data_id": data_id},
            execution_time=expiration_date
        )
```

#### 2. Implement Data Access Controls
- **DO**: Implement fine-grained access controls for data
- **DON'T**: Allow broad access to sensitive data

```python
# Example: Data access control implementation
class DataAccessControl:
    """Implement fine-grained data access controls."""
    
    def __init__(self):
        self.access_policies = load_access_policies()
        self.audit_logger = AuditLogger()
    
    def check_access(self, user, data_type, operation, context=None):
        """Check if a user has access to perform an operation on data."""
        # Get user roles
        user_roles = get_user_roles(user)
        
        # Check if any role has the required permission
        for role in user_roles:
            if self.role_has_permission(role, data_type, operation, context):
                self.audit_logger.log_access(
                    user=user,
                    data_type=data_type,
                    operation=operation,
                    context=context,
                    granted=True
                )
                return True
        
        # No role has permission
        self.audit_logger.log_access(
            user=user,
            data_type=data_type,
            operation=operation,
            context=context,
            granted=False
        )
        return False
    
    def role_has_permission(self, role, data_type, operation, context=None):
        """Check if a role has permission for an operation."""
        if role not in self.access_policies:
            return False
        
        role_policy = self.access_policies[role]
        
        # Check data type permissions
        if data_type not in role_policy:
            return False
        
        data_type_permissions = role_policy[data_type]
        
        # Check operation permissions
        if operation not in data_type_permissions["operations"]:
            return False
        
        # Check context-specific conditions if present
        if context and "conditions" in data_type_permissions:
            for condition in data_type_permissions["conditions"]:
                if not self.evaluate_condition(condition, context):
                    return False
        
        return True
    
    def evaluate_condition(self, condition, context):
        """Evaluate a condition against the context."""
        condition_type = condition["type"]
        
        if condition_type == "time_window":
            current_time = datetime.now().time()
            start_time = datetime.strptime(condition["start"], "%H:%M").time()
            end_time = datetime.strptime(condition["end"], "%H:%M").time()
            return start_time <= current_time <= end_time
        
        elif condition_type == "ip_range":
            if "ip_address" not in context:
                return False
            return ipaddress.ip_address(context["ip_address"]) in ipaddress.ip_network(condition["range"])
        
        elif condition_type == "data_owner":
            if "owner" not in context or "requester" not in context:
                return False
            return context["owner"] == context["requester"]
        
        # Add more condition types as needed
        
        return False
```

## Conclusion

Implementing these best practices will help you build robust, scalable, and responsible Generative AI applications. Remember that the field is rapidly evolving, so it's important to stay current with the latest developments and continuously refine your approach.

Key takeaways:

1. **Design for Modularity**: Separate concerns and create reusable components
2. **Prioritize Security**: Implement comprehensive security measures at all levels
3. **Optimize Performance**: Use caching, batching, and rate limiting effectively
4. **Test Thoroughly**: Implement comprehensive testing for prompts and systems
5. **Monitor Continuously**: Track technical, business, and quality metrics
6. **Handle Errors Gracefully**: Implement robust error handling and fallbacks
7. **Consider Ethics**: Address bias, transparency, and content moderation
8. **Protect Privacy**: Implement data minimization and access controls

By following these guidelines, you'll be well-positioned to create Generative AI applications that are not only technically sound but also responsible and user-focused.

## Additional Resources

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)
- [OpenAI Best Practices](https://platform.openai.com/docs/guides/safety-best-practices)
- [Responsible AI Practices](https://ai.google/responsibilities/responsible-ai-practices/)
- [Vector Database Comparison](https://www.pinecone.io/learn/vector-database-comparison/)
- [Model Evaluation Framework](https://huggingface.co/blog/evaluating-llm-responses)
