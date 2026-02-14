# Agent Squad Framework - Comprehensive Code Analysis

**Analysis Date:** February 14, 2026
**Analyzed By:** Claude Code
**Repository:** AWS Labs Agent Squad (formerly Multi-Agent Orchestrator)

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Basic Information](#basic-information)
3. [Architecture Overview](#architecture-overview)
4. [Core Modules Deep Dive](#core-modules-deep-dive)
5. [Module Interactions](#module-interactions)
6. [Advanced Features](#advanced-features)
7. [Example Use Cases](#example-use-cases)
8. [Configuration & Deployment](#configuration--deployment)
9. [Code Quality & Best Practices](#code-quality--best-practices)

---

## Executive Summary

Agent Squad is a production-ready, open-source framework for orchestrating multiple AI agents to handle complex conversations. Developed by AWS Labs, it provides intelligent intent classification, flexible response modes (streaming/non-streaming), sophisticated context management, and an extensible architecture that supports various LLM providers.

**Key Highlights:**
- Dual implementation in Python and TypeScript with full feature parity
- Intelligent classifier routes user queries to appropriate specialized agents
- Per-agent conversation history with global context awareness
- Support for AWS Bedrock, Anthropic, OpenAI, and custom agents
- Advanced features: SupervisorAgent for team coordination, RAG, tool calling, streaming
- Universal deployment: AWS Lambda, local, or any cloud platform

---

## Basic Information

| Property | Value |
|----------|-------|
| **Project Name** | Agent Squad |
| **Former Name** | Multi-Agent Orchestrator |
| **Maintainer** | AWS Labs |
| **License** | Apache 2.0 |
| **Languages** | Python 3.x, TypeScript |
| **Package Name (Python)** | `agent-squad` |
| **Package Name (TypeScript)** | `agent-squad` |
| **Repository Structure** | Dual mono-repo (python/ and typescript/ directories) |

**Authors:**
- [Corneliu Croitoru](https://www.linkedin.com/in/corneliucroitoru/)
- [Anthony Bernabeu](https://www.linkedin.com/in/anthonybernabeu/)

---

## Architecture Overview

### High-Level Flow

```
┌──────────────┐
│  User Input  │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────┐
│         AgentSquad Orchestrator         │
│  (orchestrator.py / orchestrator.ts)    │
└──┬──────────────┬───────────────┬───────┘
   │              │               │
   │ 1. Classify  │ 2. Fetch      │ 3. Process
   │              │    History    │
   ▼              ▼               ▼
┌──────────┐  ┌─────────┐   ┌──────────┐
│Classifier│  │ Storage │   │  Agents  │
└────┬─────┘  └────┬────┘   └────┬─────┘
     │             │              │
     │ Selected    │ Chat         │ Response
     │ Agent       │ History      │
     ▼             ▼              ▼
┌────────────────────────────────────────┐
│    4. Save Conversation & Return        │
└────────────────────────────────────────┘
```

### Architecture Flow Steps

1. **User Input** → User sends a message with `user_id` and `session_id`

2. **Classification Phase:**
   - Orchestrator fetches all conversation history from storage
   - Classifier analyzes user input + conversation context
   - Returns selected agent + confidence score

3. **Agent Processing Phase:**
   - Orchestrator fetches agent-specific conversation history
   - Selected agent processes request with its context
   - Agent may use tools, retrievers, or stream responses

4. **Storage & Response Phase:**
   - User message saved to storage
   - Agent response saved to storage
   - Response returned to user with metadata

---

## Core Modules Deep Dive

### 1. Orchestrator Module

**Location:** `python/src/agent_squad/orchestrator.py` | `typescript/src/orchestrator.ts`

#### Main Class: `AgentSquad`

**Responsibilities:**
- Central coordination hub for the entire framework
- Manages agent registry
- Routes requests to appropriate agents
- Coordinates classifier and storage
- Handles execution timing and logging

**Constructor Parameters:**
```python
AgentSquad(
    options: AgentSquadConfig = None,
    storage: ChatStorage = None,
    classifier: Classifier = None,
    logger: Logger = None,
    default_agent: Agent = None
)
```

**Key Properties:**
- `agents: dict[str, Agent]` - Registry of all available agents
- `classifier: Classifier` - Intent classification engine
- `storage: ChatStorage` - Conversation persistence layer
- `default_agent: Agent` - Fallback when no agent is selected
- `config: AgentSquadConfig` - Framework configuration

**Core Methods:**

1. **`add_agent(agent: Agent)`**
   - Registers a new agent with the orchestrator
   - Updates classifier with new agent information
   - Validates unique agent IDs

2. **`route_request(user_input, user_id, session_id, additional_params, stream_response)`**
   - Main entry point for processing user requests
   - Returns `AgentResponse` with metadata and output
   - Handles streaming and non-streaming modes

   **Workflow:**
   ```python
   route_request()
     ├─> classify_request()
     │     ├─> storage.fetch_all_chats()
     │     └─> classifier.classify()
     │
     ├─> agent_process_request()
     │     ├─> dispatch_to_agent()
     │     │     ├─> storage.fetch_chat(agent_id)
     │     │     └─> agent.process_request()
     │     │
     │     ├─> storage.save_message(user_message)
     │     └─> storage.save_message(agent_response)
     │
     └─> return AgentResponse
   ```

3. **`classify_request(user_input, user_id, session_id)`**
   - Fetches all conversation history across agents
   - Uses classifier to determine appropriate agent
   - Falls back to default agent if configured
   - Returns `ClassifierResult`

4. **`dispatch_to_agent(params)`**
   - Fetches agent-specific chat history
   - Calls agent's `process_request()` method
   - Measures execution time if logging enabled

**Configuration Options (`AgentSquadConfig`):**
```python
@dataclass
class AgentSquadConfig:
    LOG_AGENT_CHAT: bool = False
    LOG_CLASSIFIER_CHAT: bool = False
    LOG_CLASSIFIER_RAW_OUTPUT: bool = False
    LOG_CLASSIFIER_OUTPUT: bool = False
    LOG_EXECUTION_TIMES: bool = False
    MAX_RETRIES: int = 3
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED: bool = True
    MAX_MESSAGE_PAIRS_PER_AGENT: int = 100
    NO_SELECTED_AGENT_MESSAGE: str
    GENERAL_ROUTING_ERROR_MSG_MESSAGE: str
```

---

### 2. Agents Module

**Location:** `python/src/agent_squad/agents/` | `typescript/src/agents/`

#### Base Agent Class

**File:** `agent.py` | `agent.ts`

```python
class Agent(ABC):
    def __init__(self, options: AgentOptions):
        self.name: str
        self.id: str  # Auto-generated from name
        self.description: str
        self.save_chat: bool = True
        self.callbacks: AgentCallbacks
        self.log_debug_trace: bool

    @abstractmethod
    async def process_request(
        input_text: str,
        user_id: str,
        session_id: str,
        chat_history: list[ConversationMessage],
        additional_params: dict
    ) -> ConversationMessage | AsyncIterable[AgentOutputType]

    def is_streaming_enabled() -> bool:
        return False  # Override in subclasses

    @staticmethod
    def generate_key_from_name(name: str) -> str:
        # Converts "Tech Agent" -> "tech-agent"
```

**Agent Lifecycle Callbacks:**
```python
class AgentCallbacks:
    async def on_agent_start(agent_name, payload_input, messages, run_id, tags, metadata, **kwargs)
    async def on_agent_end(agent_name, response, messages, run_id, tags, metadata, **kwargs)
    async def on_llm_start(name, payload_input, run_id, tags, metadata, **kwargs)
    async def on_llm_new_token(token, **kwargs)
    async def on_llm_end(name, output, run_id, tags, metadata, **kwargs)
```

#### Built-in Agent Implementations

##### 1. BedrockLLMAgent

**File:** `bedrock_llm_agent.py`

**Purpose:** AWS Bedrock integration using Converse API

**Key Features:**
- Streaming and non-streaming support
- Tool/function calling with automatic recursion
- Retriever integration for RAG
- Guardrail configuration
- Thinking mode support (extended thinking/reasoning)
- Custom system prompts with variable substitution

**Configuration:**
```python
@dataclass
class BedrockLLMAgentOptions(AgentOptions):
    model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"
    region: str = None
    streaming: bool = False
    inference_config: dict = {
        "maxTokens": 1000,
        "temperature": 0.0,
        "topP": 0.9,
        "stopSequences": []
    }
    guardrail_config: dict = None
    retriever: Retriever = None
    tool_config: dict | AgentTools = None
    custom_system_prompt: dict = None
    client: boto3.client = None
    additional_model_request_fields: dict = None
```

**Processing Flow:**
```python
process_request()
  ├─> _prepare_system_prompt()
  │     └─> retriever.retrieve_and_combine_results() [if configured]
  │
  ├─> _prepare_conversation()
  │     └─> Append user message to chat_history
  │
  ├─> _build_conversation_command()
  │     ├─> Set model_id, messages, system, inferenceConfig
  │     ├─> Add guardrailConfig [if configured]
  │     ├─> Add toolConfig [if configured]
  │     └─> Add additionalModelRequestFields [if configured]
  │
  └─> _process_with_strategy()
        ├─> STREAMING:
        │     └─> _handle_streaming()
        │           ├─> client.converse_stream()
        │           ├─> Yield AgentStreamResponse chunks
        │           ├─> Check for toolUse
        │           ├─> Execute tools if needed
        │           └─> Recurse until no more tools
        │
        └─> NON-STREAMING:
              └─> _handle_single_response_loop()
                    ├─> client.converse()
                    ├─> Check for toolUse
                    ├─> Execute tools if needed
                    └─> Recurse until no more tools
```

**Tool Handling:**
- Supports `AgentTools` class with automatic tool execution
- Custom tool handler via `useToolHandler` callback
- Maximum recursions configurable (default: 20)
- Tools can be called in chains until final text response

**Thinking Mode:**
```python
additional_model_request_fields = {
    "thinking": {"type": "enabled"}
}
```
- Streams reasoning content separately from regular text
- `AgentStreamResponse.thinking` contains reasoning chunks
- `AgentStreamResponse.final_thinking` contains complete reasoning
- Callbacks receive thinking tokens with `thinking=True` flag

##### 2. SupervisorAgent

**File:** `supervisor_agent.py`

**Purpose:** Coordinates a team of specialized agents (agent-as-tools pattern)

**Key Concept:**
The SupervisorAgent enables hierarchical multi-agent systems where a lead agent coordinates a team of specialized agents, treating each team member as a tool it can invoke.

**Architecture:**
```python
@dataclass
class SupervisorAgentOptions(AgentOptions):
    lead_agent: BedrockLLMAgent | AnthropicAgent  # Coordinator
    team: list[Agent]  # Specialized agents
    storage: ChatStorage = None  # Separate memory for team
    trace: bool = False  # Enable logging
    extra_tools: AgentTools | list[AgentTool] = None  # Additional tools
```

**Built-in Tool:**
```python
{
    "name": "send_messages",
    "description": "Send messages to multiple agents in parallel.",
    "properties": {
        "messages": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "recipient": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["recipient", "content"]
            }
        }
    }
}
```

**How It Works:**

1. **Setup Phase:**
   - Lead agent gets `send_messages` tool + any extra_tools
   - System prompt configured with team member descriptions
   - Separate storage maintains per-team-member history

2. **Execution Phase:**
   ```python
   User: "Create a marketing campaign for eco-friendly shoes"

   Lead Agent (Coordinator):
     ├─> Analyzes request
     ├─> Decides which team members to contact
     └─> Uses send_messages tool:
           {
             "messages": [
               {"recipient": "Market Researcher", "content": "Research eco-shoe market"},
               {"recipient": "Copywriter", "content": "Write tagline for eco shoes"},
               {"recipient": "Designer", "content": "Create visual concept"}
             ]
           }

   send_messages() executes:
     ├─> Contacts Market Researcher in parallel
     ├─> Contacts Copywriter in parallel
     ├─> Contacts Designer in parallel
     └─> Returns aggregated results

   Lead Agent:
     ├─> Receives all responses
     ├─> Synthesizes final campaign
     └─> Returns to user
   ```

3. **Parallel Processing:**
   ```python
   async def send_messages(self, messages: list[dict]):
       tasks = [
           self.send_message(agent, msg['content'], ...)
           for msg in messages
           for agent in self.team
           if agent.name == msg['recipient']
       ]
       results = await asyncio.gather(*tasks)
       return aggregated_results
   ```

**System Prompt Guidelines:**
- Don't mention agent names to user
- Contact multiple agents simultaneously when possible
- Act as intermediary between agents (they can't see each other)
- Don't summarize agent responses
- Forward yes/no answers directly to last agent
- Reuse cached responses from agent memory

**Use Cases:**
- AI Production Studio (director + writer + cinematographer + editor)
- Travel Planning (flights + hotels + activities + insurance)
- Customer Support Teams (billing + tech + orders + returns)
- Healthcare Coordination (doctor + pharmacy + insurance + labs)

##### 3. Other Agent Types

**AmazonBedrockAgent** (`amazon_bedrock_agent.py`)
- Pre-configured AWS Bedrock Agents
- Invokes existing Bedrock Agent resources
- Manages agent aliases and versions

**AnthropicAgent** (`anthropic_agent.py`)
- Direct Anthropic API integration
- Claude models (Haiku, Sonnet, Opus)
- Streaming support
- Tool calling

**OpenAIAgent** (`openai_agent.py`)
- OpenAI GPT models
- Function calling
- Streaming support

**LexBotAgent** (`lex_bot_agent.py`)
- Amazon Lex integration
- Voice and text chatbots
- Intent-based interactions

**LambdaAgent** (`lambda_agent.py`)
- AWS Lambda function as agent
- Custom business logic
- Serverless execution

**BedrockFlowsAgent** (`bedrock_flows_agent.py`)
- Amazon Bedrock Flows integration
- Workflow orchestration

**BedrockInlineAgent** (`bedrock_inline_agent.py`)
- Dynamic inline agent creation
- Runtime knowledge base selection

**BedrockTranslatorAgent** (`bedrock_translator_agent.py`)
- Multi-language translation
- Enables multilingual chatbots

**ChainAgent** (`chain_agent.py`)
- Sequential agent execution
- Pipeline processing

**ComprehendFilterAgent** (`comprehend_filter_agent.py`)
- AWS Comprehend integration
- Content filtering and sentiment analysis

---

### 3. Classifiers Module

**Location:** `python/src/agent_squad/classifiers/` | `typescript/src/classifiers/`

#### Base Classifier Class

**File:** `classifier.py` | `classifier.ts`

**Purpose:** Intent classification and agent selection engine

```python
class Classifier(ABC):
    def __init__(self):
        self.agent_descriptions: str
        self.history: str
        self.custom_variables: TemplateVariables
        self.prompt_template: str
        self.system_prompt: str
        self.agents: dict[str, Agent]

    def set_agents(self, agents: dict[str, Agent]):
        # Updates agent descriptions for classification

    def set_history(self, messages: list[ConversationMessage]):
        # Formats conversation history for prompt

    async def classify(input_text: str, chat_history: list[ConversationMessage]) -> ClassifierResult:
        # Main classification method

    @abstractmethod
    async def process_request(input_text, chat_history) -> ClassifierResult
```

**Classification Prompt Template:**

The built-in prompt is sophisticated and handles:

1. **Initial Queries:**
   ```
   User: "What are the symptoms of the flu?"
   → Selects HealthAgent with high confidence
   ```

2. **Follow-up Responses:**
   ```
   Previous: TechAgent helped with printer setup
   User: "Yes, please give me detailed instructions"
   → Selects same TechAgent (continuation)
   ```

3. **Context Switching:**
   ```
   Previous: TechAgent
   User: "Actually, I need to know about my account balance"
   → Switches to BillingAgent
   ```

4. **Multi-turn with Context:**
   ```
   Turn 1: User asks about pricing → BillingAgent
   Turn 2: User asks about refunds → BillingAgent
   Turn 3: User has login issue → TechAgent (context switch)
   Turn 4: User says "It says password incorrect" → TechAgent (continuation)
   ```

**Classification Guidelines:**
- **Priority Assignment:** High (urgent), Medium (standard), Low (general)
- **Key Entities:** Extract important nouns, products, issues
- **Confidence Levels:** High (clear), Medium (some ambiguity), Low (vague)
- **Follow-up Detection:** Recognize "yes", "ok", "tell me more", numbers, etc.

**Prompt Variables:**
- `{{AGENT_DESCRIPTIONS}}` - All agent descriptions
- `{{HISTORY}}` - Formatted conversation history
- Custom variables via `set_system_prompt()`

#### Classifier Implementations

##### BedrockClassifier

**File:** `bedrock_classifier.py`

```python
@dataclass
class BedrockClassifierOptions:
    model_id: str = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    inference_config: dict = {
        "maxTokens": 500,
        "temperature": 0.0,
        "topP": 0.9
    }
    client: boto3.client = None
    region: str = None

class BedrockClassifier(Classifier):
    async def process_request(input_text, chat_history) -> ClassifierResult:
        # Uses Bedrock Converse API
        # Parses LLM response for agent selection
        # Returns ClassifierResult
```

**Response Parsing:**
```python
# Expected LLM output format:
"""
userinput: What's the weather in Seattle?
selected_agent: weather-agent
confidence: 0.95
"""

# Extracts:
# - selected_agent ID
# - Confidence score
# - Maps to Agent object
```

##### AnthropicClassifier

**File:** `anthropic_classifier.py`

Similar to BedrockClassifier but uses Anthropic API directly.

##### OpenAIClassifier

**File:** `openai_classifier.py`

Uses OpenAI models for classification.

**ClassifierResult Data Structure:**
```python
@dataclass
class ClassifierResult:
    selected_agent: Optional[Agent]  # None if no match
    confidence: float  # 0.0 to 1.0
```

---

### 4. Storage Module

**Location:** `python/src/agent_squad/storage/` | `typescript/src/storage/`

#### Base Storage Interface

**File:** `chat_storage.py` | `chatStorage.ts`

```python
class ChatStorage(ABC):
    @abstractmethod
    async def save_chat_message(
        user_id: str,
        session_id: str,
        agent_id: str,
        new_message: ConversationMessage | TimestampedMessage,
        max_history_size: int = None
    ) -> bool

    @abstractmethod
    async def save_chat_messages(
        user_id: str,
        session_id: str,
        agent_id: str,
        new_messages: list[ConversationMessage | TimestampedMessage],
        max_history_size: int = None
    ) -> bool

    @abstractmethod
    async def fetch_chat(
        user_id: str,
        session_id: str,
        agent_id: str,
        max_history_size: int = None
    ) -> list[ConversationMessage]

    @abstractmethod
    async def fetch_all_chats(
        user_id: str,
        session_id: str
    ) -> list[ConversationMessage]

    # Helper methods
    def is_same_role_as_last_message(conversation, new_message) -> bool
    def trim_conversation(conversation, max_history_size) -> list
```

**Key Concepts:**

1. **Composite Key:** `user_id#session_id#agent_id`
   - Separates conversations per user
   - Isolates sessions
   - Maintains per-agent history

2. **Timestamped Messages:**
   ```python
   class TimestampedMessage(ConversationMessage):
       timestamp: int  # milliseconds since epoch
   ```

3. **Conversation Trimming:**
   - Keeps last N message pairs (user + assistant)
   - Ensures even number for complete exchanges
   - Prevents unbounded memory growth

4. **Consecutive Message Prevention:**
   - Blocks saving consecutive messages from same role
   - Prevents "assistant → assistant" or "user → user"

#### InMemoryChatStorage

**File:** `in_memory_chat_storage.py`

**Implementation:**
```python
class InMemoryChatStorage(ChatStorage):
    def __init__(self):
        self.conversations = defaultdict(list)
        # Structure: {
        #   "user1#session1#agent1": [TimestampedMessage, ...],
        #   "user1#session1#agent2": [TimestampedMessage, ...],
        #   "user2#session1#agent1": [TimestampedMessage, ...]
        # }

    async def save_chat_message(...):
        key = f"{user_id}#{session_id}#{agent_id}"

        # Prevent consecutive same-role messages
        if self.is_same_role_as_last_message(conversation, new_message):
            return conversation

        # Add timestamp
        timestamped = TimestampedMessage(
            role=new_message.role,
            content=new_message.content,
            timestamp=int(time.time() * 1000)
        )

        # Append and trim
        conversation.append(timestamped)
        conversation = self.trim_conversation(conversation, max_history_size)
        self.conversations[key] = conversation

        return self._remove_timestamps(conversation)

    async def fetch_all_chats(user_id, session_id):
        all_messages = []

        # Aggregate from all agents
        for key, messages in self.conversations.items():
            stored_user_id, stored_session_id, agent_id = key.split('#')
            if stored_user_id == user_id and stored_session_id == session_id:
                # Prefix agent responses with [agent_id]
                for message in messages:
                    if message.role == "assistant":
                        content = [{'text': f"[{agent_id}] {message.content[0]['text']}"}]
                    else:
                        content = message.content
                    all_messages.append(TimestampedMessage(...))

        # Sort by timestamp
        all_messages.sort(key=lambda x: x.timestamp)
        return all_messages
```

**Why Prefix Agent ID in fetch_all_chats:**
```
Classifier sees:
  User: "Book a flight to NYC"
  Assistant: "[travel-agent] I can help you book a flight..."
  User: "Yes, for next Monday"
  Assistant: "[travel-agent] Great, I'll search for flights on Monday..."
  User: "Actually, how much will this cost?"
  Assistant: "[billing-agent] Let me check pricing for you..."

This context helps classifier understand:
- Which agent handled which response
- Whether user is continuing with same agent
- When context switches occurred
```

#### DynamoDBChatStorage

**File:** `dynamodb_chat_storage.py`

**Features:**
- Persistent storage in AWS DynamoDB
- Table schema:
  ```
  PK: user_id#session_id
  SK: agent_id#timestamp
  conversation: [messages]
  ```
- TTL support for automatic cleanup
- Conditional writes to prevent race conditions

#### SQLChatStorage

**File:** `sql_chat_storage.py`

**Features:**
- SQL database storage (MySQL, PostgreSQL, SQLite)
- Schema:
  ```sql
  CREATE TABLE conversations (
      user_id VARCHAR(255),
      session_id VARCHAR(255),
      agent_id VARCHAR(255),
      timestamp BIGINT,
      role VARCHAR(50),
      content TEXT,
      PRIMARY KEY (user_id, session_id, agent_id, timestamp)
  )
  ```

---

### 5. Retrievers Module

**Location:** `python/src/agent_squad/retrievers/` | `typescript/src/retrievers/`

#### Base Retriever Interface

**File:** `retriever.py` | `retriever.ts`

```python
class Retriever(ABC):
    @abstractmethod
    async def retrieve_and_combine_results(query: str) -> str:
        """
        Retrieve relevant information and combine into a string.
        This string is appended to the agent's system prompt.
        """
        pass
```

#### AmazonKBRetriever

**File:** `amazon_kb_retriever.py`

**Purpose:** Integration with Amazon Bedrock Knowledge Bases

```python
class AmazonKBRetriever(Retriever):
    def __init__(
        self,
        knowledge_base_id: str,
        region: str = "us-east-1",
        max_results: int = 5
    ):
        self.kb_id = knowledge_base_id
        self.client = boto3.client('bedrock-agent-runtime', region_name=region)
        self.max_results = max_results

    async def retrieve_and_combine_results(self, query: str) -> str:
        response = self.client.retrieve(
            knowledgeBaseId=self.kb_id,
            retrievalQuery={'text': query},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': self.max_results
                }
            }
        )

        # Combine retrieved chunks
        chunks = []
        for result in response['retrievalResults']:
            chunks.append(result['content']['text'])

        return "\n\n".join(chunks)
```

**Usage with Agent:**
```python
retriever = AmazonKBRetriever(
    knowledge_base_id="ABCDEF123456",
    region="us-east-1",
    max_results=5
)

agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Documentation Agent",
    description="Answers questions about product documentation",
    retriever=retriever
))

# When agent processes request:
# 1. query = "How do I reset my password?"
# 2. retriever.retrieve_and_combine_results(query)
# 3. Retrieved context appended to system prompt:
#    "Here is the context to use to answer the user's question:
#     [retrieved documentation chunks]"
# 4. LLM generates response with context
```

---

### 6. Types Module

**Location:** `python/src/agent_squad/types/types.py` | `typescript/src/types/index.ts`

#### Core Data Structures

**ConversationMessage:**
```python
class ConversationMessage:
    role: ParticipantRole  # "user" or "assistant"
    content: list[Any]     # List of content blocks

    # Content block examples:
    # [{"text": "Hello"}]
    # [{"text": "Weather is"}, {"toolUse": {...}}]
    # [{"reasoningContent": {...}}, {"text": "Answer"}]
```

**TimestampedMessage:**
```python
class TimestampedMessage(ConversationMessage):
    timestamp: int  # Milliseconds since epoch

    def __init__(self, role, content, timestamp=None):
        super().__init__(role, content)
        self.timestamp = timestamp or int(time.time() * 1000)
```

**AgentProcessingResult:**
```python
@dataclass
class AgentProcessingResult:
    user_input: str        # Original user query
    agent_id: str          # Selected agent ID
    agent_name: str        # Selected agent name
    user_id: str           # User identifier
    session_id: str        # Session identifier
    additional_params: dict  # Extra metadata
```

**AgentStreamResponse:**
```python
@dataclass
class AgentStreamResponse:
    text: str = ""                                    # Regular text chunk
    thinking: str = ""                                 # Reasoning chunk
    final_message: ConversationMessage = None          # Complete message
    final_thinking: str = None                         # Complete reasoning
```

**AgentResponse:**
```python
@dataclass
class AgentResponse:
    metadata: AgentProcessingResult
    output: ConversationMessage | AsyncIterable[AgentStreamResponse]
    streaming: bool
```

**AgentSquadConfig:**
```python
@dataclass
class AgentSquadConfig:
    LOG_AGENT_CHAT: bool = False
    LOG_CLASSIFIER_CHAT: bool = False
    LOG_CLASSIFIER_RAW_OUTPUT: bool = False
    LOG_CLASSIFIER_OUTPUT: bool = False
    LOG_EXECUTION_TIMES: bool = False
    MAX_RETRIES: int = 3
    USE_DEFAULT_AGENT_IF_NONE_IDENTIFIED: bool = True
    MAX_MESSAGE_PAIRS_PER_AGENT: int = 100
    NO_SELECTED_AGENT_MESSAGE: str
    GENERAL_ROUTING_ERROR_MSG_MESSAGE: str
```

**Enums:**
```python
class ParticipantRole(Enum):
    ASSISTANT = "assistant"
    USER = "user"

class AgentProviderType(Enum):
    BEDROCK = "BEDROCK"
    ANTHROPIC = "ANTHROPIC"
```

---

### 7. Utils Module

**Location:** `python/src/agent_squad/utils/` | `typescript/src/utils/`

#### AgentTool & AgentTools

**File:** `tool.py` | `tool.ts`

**Purpose:** Tool/function calling infrastructure

```python
class AgentTool:
    def __init__(
        self,
        name: str,
        description: str,
        properties: dict,
        required: list[str] = [],
        func: Callable = None,
        func_description: str = None
    ):
        self.name = name
        self.description = description
        self.properties = properties
        self.required = required
        self.func = func
        self.func_description = func_description or description

    def to_bedrock_format(self) -> dict:
        return {
            "toolSpec": {
                "name": self.name,
                "description": self.description,
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": self.properties,
                        "required": self.required
                    }
                }
            }
        }

    async def execute(self, tool_input: dict, **kwargs) -> Any:
        return await self.func(tool_input, **kwargs)

class AgentTools:
    def __init__(
        self,
        tools: list[AgentTool],
        callbacks: ToolCallbacks = None
    ):
        self.tools = tools
        self.callbacks = callbacks or ToolCallbacks()

    def to_bedrock_format(self) -> list[dict]:
        return [tool.to_bedrock_format() for tool in self.tools]

    async def tool_handler(
        self,
        provider: str,  # "BEDROCK" or "ANTHROPIC"
        llm_response: ConversationMessage,
        conversation: list[ConversationMessage],
        additional_params: dict = None
    ) -> ConversationMessage:
        """
        Executes tools and returns tool results message
        """
        tool_results = []

        for content in llm_response.content:
            if "toolUse" in content:
                tool_use = content["toolUse"]
                tool_name = tool_use["name"]
                tool_input = tool_use["input"]

                # Find and execute tool
                tool = next(t for t in self.tools if t.name == tool_name)
                result = await tool.execute(tool_input, **additional_params)

                # Callback
                await self.callbacks.on_tool_execution(tool_name, tool_input, result)

                # Format result
                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"text": json.dumps(result)}]
                    }
                })

        return ConversationMessage(
            role="user",
            content=tool_results
        )
```

**Example Tool Definition:**
```python
def get_weather(tool_input: dict, **kwargs) -> dict:
    location = tool_input['location']
    # Call weather API
    return {
        "location": location,
        "temperature": 72,
        "conditions": "sunny"
    }

weather_tool = AgentTool(
    name="get_weather",
    description="Get current weather for a location",
    properties={
        "location": {
            "type": "string",
            "description": "City name"
        }
    },
    required=["location"],
    func=get_weather
)

tools = AgentTools([weather_tool])

agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Weather Agent",
    description="Provides weather information",
    tool_config={
        "tool": tools,
        "toolMaxRecursions": 10
    }
))
```

#### Logger

**File:** `logger.py` | `logger.ts`

**Purpose:** Structured logging with configuration

```python
class Logger:
    @staticmethod
    def info(message: str):
        print(f"[INFO] {message}")

    @staticmethod
    def warn(message: str):
        print(f"[WARN] {message}")

    @staticmethod
    def error(message: str):
        print(f"[ERROR] {message}")

    @staticmethod
    def debug(message: str):
        print(f"[DEBUG] {message}")
```

#### Helpers

**File:** `helpers.py` | `helpers.ts`

```python
def conversation_to_dict(conversation: list[ConversationMessage]) -> list[dict]:
    """
    Convert ConversationMessage objects to dictionary format for API calls
    """
    return [
        {
            "role": msg.role,
            "content": msg.content
        }
        for msg in conversation
    ]
```

---

## Module Interactions

### Complete Request Flow with Code

```python
# 1. User initiates request
orchestrator = AgentSquad(
    storage=InMemoryChatStorage(),
    classifier=BedrockClassifier()
)

# 2. Register agents
orchestrator.add_agent(tech_agent)
orchestrator.add_agent(billing_agent)
orchestrator.add_agent(travel_agent)

# 3. Route request
response = await orchestrator.route_request(
    user_input="I need help with my account",
    user_id="user_123",
    session_id="session_456"
)
```

**Internal Flow:**

```python
# orchestrator.py - route_request()
async def route_request(user_input, user_id, session_id, ...):

    # Step 1: Classify
    classifier_result = await self.classify_request(user_input, user_id, session_id)
    # classifier_result = ClassifierResult(
    #     selected_agent=billing_agent,
    #     confidence=0.92
    # )

    # Step 2: Process with agent
    return await self.agent_process_request(
        user_input, user_id, session_id, classifier_result, ...
    )

# orchestrator.py - classify_request()
async def classify_request(user_input, user_id, session_id):

    # Fetch ALL conversation history (from all agents)
    chat_history = await self.storage.fetch_all_chats(user_id, session_id)
    # chat_history = [
    #     ConversationMessage(role="user", content=[{"text": "Book flight"}]),
    #     ConversationMessage(role="assistant", content=[{"text": "[travel-agent] Where to?"}]),
    #     ConversationMessage(role="user", content=[{"text": "Seattle"}]),
    #     ConversationMessage(role="assistant", content=[{"text": "[travel-agent] Flight booked"}]),
    #     ConversationMessage(role="user", content=[{"text": "I need help with my account"}])
    # ]

    # Classify with context
    return await self.classifier.classify(user_input, chat_history)

# classifier.py - classify()
async def classify(input_text, chat_history):
    self.set_history(chat_history)
    self.update_system_prompt()
    # system_prompt now contains:
    # - Agent descriptions
    # - Formatted chat history
    # - Classification instructions

    return await self.process_request(input_text, chat_history)

# bedrock_classifier.py - process_request()
async def process_request(input_text, chat_history):
    response = self.client.converse(
        modelId=self.model_id,
        messages=[{"role": "user", "content": [{"text": input_text}]}],
        system=[{"text": self.system_prompt}]
    )

    # Parse response
    output = response["output"]["message"]["content"][0]["text"]
    # "userinput: I need help with my account
    #  selected_agent: billing-agent
    #  confidence: 0.92"

    agent_id = extract_agent_id(output)
    confidence = extract_confidence(output)

    return ClassifierResult(
        selected_agent=self.get_agent_by_id(agent_id),
        confidence=confidence
    )

# orchestrator.py - agent_process_request()
async def agent_process_request(user_input, user_id, session_id, classifier_result, ...):
    selected_agent = classifier_result.selected_agent

    # Fetch agent-specific history
    agent_chat_history = await self.storage.fetch_chat(
        user_id, session_id, selected_agent.id
    )
    # agent_chat_history = [] (first time with billing agent)

    # Dispatch to agent
    agent_response = await selected_agent.process_request(
        input_text=user_input,
        user_id=user_id,
        session_id=session_id,
        chat_history=agent_chat_history,
        additional_params={}
    )

    # Save user message
    await self.storage.save_chat_message(
        user_id, session_id, selected_agent.id,
        ConversationMessage(role="user", content=[{"text": user_input}])
    )

    # Save agent response
    if not streaming:
        await self.storage.save_chat_message(
            user_id, session_id, selected_agent.id,
            agent_response
        )

    return AgentResponse(
        metadata=AgentProcessingResult(...),
        output=agent_response,
        streaming=selected_agent.is_streaming_enabled()
    )

# bedrock_llm_agent.py - process_request()
async def process_request(input_text, user_id, session_id, chat_history, ...):

    # Prepare system prompt (with retriever if configured)
    system_prompt = await self._prepare_system_prompt(input_text)

    # Build conversation
    conversation = [
        *chat_history,
        ConversationMessage(role="user", content=[{"text": input_text}])
    ]

    # Build API command
    command = {
        "modelId": self.model_id,
        "messages": conversation_to_dict(conversation),
        "system": [{"text": system_prompt}],
        "inferenceConfig": self.inference_config
    }

    if self.tool_config:
        command["toolConfig"] = self._prepare_tool_config()

    # Process with streaming or non-streaming
    if self.streaming:
        return self._handle_streaming(command, conversation, max_recursions=20)
    else:
        return await self._handle_single_response_loop(command, conversation, max_recursions=20)

# bedrock_llm_agent.py - _handle_single_response_loop()
async def _handle_single_response_loop(command, conversation, max_recursions):

    while max_recursions > 0:
        # Call LLM
        llm_response = await self.handle_single_response(command)
        # llm_response = ConversationMessage(
        #     role="assistant",
        #     content=[{"text": "I can help you with your account..."}]
        # )

        conversation.append(llm_response)

        # Check for tool use
        has_tools = any("toolUse" in content for content in llm_response.content)

        if has_tools:
            # Execute tools
            tool_response = await self._process_tool_block(llm_response, conversation)
            conversation.append(tool_response)
            command["messages"] = conversation_to_dict(conversation)
            max_recursions -= 1
        else:
            break

    return llm_response
```

### Data Flow Visualization

```
Request: "I need help with my account"
User: user_123
Session: session_456

┌──────────────────────────────────────────────────────────────┐
│ 1. Orchestrator.route_request()                              │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Storage.fetch_all_chats(user_123, session_456)            │
│    Returns: [                                                 │
│      {role: "user", content: "Book flight"},                  │
│      {role: "assistant", content: "[travel-agent] Where?"},   │
│      {role: "user", content: "Seattle"},                      │
│      {role: "assistant", content: "[travel-agent] Booked"}    │
│    ]                                                          │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Classifier.classify(input, history)                        │
│    Analyzes:                                                  │
│    - User input: "I need help with my account"                │
│    - History: Previous interaction with travel-agent          │
│    - Agent descriptions: tech-agent, billing-agent, travel    │
│    Returns: ClassifierResult(                                 │
│      selected_agent=billing-agent,                            │
│      confidence=0.92                                          │
│    )                                                          │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Storage.fetch_chat(user_123, session_456, billing-agent)  │
│    Returns: [] (first time with billing agent)                │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. BillingAgent.process_request(                             │
│      input="I need help with my account",                     │
│      chat_history=[]                                          │
│    )                                                          │
│    ├─> Prepares system prompt                                │
│    ├─> Builds conversation: [                                │
│    │     {role: "user", content: "I need help..."}            │
│    │   ]                                                      │
│    ├─> Calls LLM (Bedrock Converse)                          │
│    └─> Returns: ConversationMessage(                         │
│          role="assistant",                                    │
│          content="I can help with your account..."            │
│        )                                                      │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. Storage.save_chat_message() x2                            │
│    - Save user message                                        │
│    - Save agent response                                      │
│    Storage now contains:                                      │
│    {                                                          │
│      "user_123#session_456#travel-agent": [...],              │
│      "user_123#session_456#billing-agent": [                  │
│        {role: "user", content: "I need help..."},             │
│        {role: "assistant", content: "I can help..."}          │
│      ]                                                        │
│    }                                                          │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. Return AgentResponse(                                     │
│      metadata={                                               │
│        user_input: "I need help with my account",             │
│        agent_id: "billing-agent",                             │
│        agent_name: "Billing Agent",                           │
│        user_id: "user_123",                                   │
│        session_id: "session_456"                              │
│      },                                                       │
│      output=ConversationMessage(...),                         │
│      streaming=False                                          │
│    )                                                          │
└──────────────────────────────────────────────────────────────┘
```

---

## Advanced Features

### 1. Streaming Responses

**Implementation:**

```python
# Agent configuration
agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Tech Support",
    description="Technical support agent",
    streaming=True  # Enable streaming
))

# Request with streaming
response = await orchestrator.route_request(
    "How do I reset my password?",
    user_id="user_123",
    session_id="session_456",
    stream_response=True  # Request streaming output
)

# Handle streaming response
if response.streaming:
    async for chunk in response.output:
        if isinstance(chunk, AgentStreamResponse):
            # Regular text
            if chunk.text:
                print(chunk.text, end='', flush=True)

            # Thinking (if enabled)
            if chunk.thinking:
                print(f"\n[Thinking: {chunk.thinking}]", flush=True)

            # Final message with complete content
            if chunk.final_message:
                # Save to database, process metadata, etc.
                full_text = chunk.final_message.content[0]['text']
```

**Streaming Flow:**

```python
# bedrock_llm_agent.py
async def _handle_streaming(command, conversation, max_recursions):
    async def stream_generator():
        while max_recursions > 0:
            response = self.handle_streaming_response(command)

            async for chunk in response:
                # Yield each chunk as it arrives
                yield chunk

                if chunk.final_message:
                    final_response = chunk.final_message

            # Check for tools
            if has_tool_use(final_response):
                tool_response = await execute_tools(final_response)
                conversation.append(tool_response)
                command["messages"] = conversation_to_dict(conversation)
                max_recursions -= 1
            else:
                break

    return stream_generator()

async def handle_streaming_response(converse_input):
    response = self.client.converse_stream(**converse_input)

    accumulated_text = ""
    accumulated_thinking = ""

    for chunk in response["stream"]:
        if "contentBlockDelta" in chunk:
            delta = chunk["contentBlockDelta"]["delta"]

            # Text chunk
            if "text" in delta:
                accumulated_text += delta["text"]
                yield AgentStreamResponse(text=delta["text"])

            # Thinking chunk
            elif "reasoningContent" in delta:
                if "text" in delta["reasoningContent"]:
                    thinking_text = delta["reasoningContent"]["text"]
                    accumulated_thinking += thinking_text
                    yield AgentStreamResponse(thinking=thinking_text)

    # Final message
    final_message = ConversationMessage(
        role="assistant",
        content=[
            {"reasoningContent": accumulated_thinking} if accumulated_thinking else None,
            {"text": accumulated_text}
        ]
    )

    yield AgentStreamResponse(
        final_message=final_message,
        final_thinking=accumulated_thinking
    )
```

### 2. Tool/Function Calling

**Complete Example:**

```python
# Define tool function
async def get_current_weather(tool_input: dict, **kwargs) -> dict:
    location = tool_input['location']
    unit = tool_input.get('unit', 'fahrenheit')

    # Call weather API
    weather_data = fetch_weather_api(location)

    return {
        "location": location,
        "temperature": weather_data['temp'],
        "unit": unit,
        "conditions": weather_data['conditions'],
        "forecast": weather_data['forecast']
    }

async def search_flights(tool_input: dict, **kwargs) -> dict:
    origin = tool_input['origin']
    destination = tool_input['destination']
    date = tool_input['date']

    # Call flight API
    flights = search_flight_api(origin, destination, date)

    return {
        "flights": flights,
        "count": len(flights)
    }

# Create tools
tools = AgentTools([
    AgentTool(
        name="get_current_weather",
        description="Get the current weather in a given location",
        properties={
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit"
            }
        },
        required=["location"],
        func=get_current_weather
    ),
    AgentTool(
        name="search_flights",
        description="Search for available flights",
        properties={
            "origin": {"type": "string", "description": "Departure city"},
            "destination": {"type": "string", "description": "Arrival city"},
            "date": {"type": "string", "description": "Departure date (YYYY-MM-DD)"}
        },
        required=["origin", "destination", "date"],
        func=search_flights
    )
])

# Create agent with tools
agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Travel Assistant",
    description="Helps with travel planning, weather, and flights",
    tool_config={
        "tool": tools,
        "toolMaxRecursions": 10
    }
))

# Usage
orchestrator = AgentSquad()
orchestrator.add_agent(agent)

response = await orchestrator.route_request(
    "What's the weather in Seattle and find me flights from SF to Seattle for next Monday",
    user_id="user_123",
    session_id="session_456"
)
```

**Tool Execution Flow:**

```
User: "What's the weather in Seattle and find flights from SF to Seattle for next Monday"

LLM Response 1:
  content: [
    {
      "toolUse": {
        "toolUseId": "tool_1",
        "name": "get_current_weather",
        "input": {"location": "Seattle, WA"}
      }
    },
    {
      "toolUse": {
        "toolUseId": "tool_2",
        "name": "search_flights",
        "input": {
          "origin": "San Francisco, CA",
          "destination": "Seattle, WA",
          "date": "2026-02-21"
        }
      }
    }
  ]

Tool Execution:
  ├─> Execute get_current_weather({"location": "Seattle, WA"})
  │   Returns: {"temperature": 55, "unit": "fahrenheit", "conditions": "cloudy"}
  │
  └─> Execute search_flights({...})
      Returns: {"flights": [...], "count": 5}

Tool Results Message:
  role: "user"
  content: [
    {
      "toolResult": {
        "toolUseId": "tool_1",
        "content": [{"text": '{"temperature": 55, "conditions": "cloudy"}'}]
      }
    },
    {
      "toolResult": {
        "toolUseId": "tool_2",
        "content": [{"text": '{"flights": [...], "count": 5}'}]
      }
    }
  ]

LLM Response 2 (with tool results):
  content: [
    {
      "text": "The weather in Seattle is currently 55°F and cloudy. I found 5 flights from San Francisco to Seattle on Monday, February 21st. The earliest flight departs at 6:30 AM..."
    }
  ]

Final Response to User:
  "The weather in Seattle is currently 55°F and cloudy. I found 5 flights..."
```

**Custom Tool Handler:**

```python
async def custom_tool_handler(
    llm_response: ConversationMessage,
    conversation: list[ConversationMessage]
) -> ConversationMessage:
    """
    Custom tool execution logic
    """
    tool_results = []

    for content in llm_response.content:
        if "toolUse" in content:
            tool_use = content["toolUse"]
            tool_name = tool_use["name"]
            tool_input = tool_use["input"]

            # Custom logic
            if tool_name == "special_tool":
                result = await special_processing(tool_input)
            else:
                result = await default_processing(tool_input)

            # Add to results
            tool_results.append({
                "toolResult": {
                    "toolUseId": tool_use["toolUseId"],
                    "content": [{"text": json.dumps(result)}]
                }
            })

    return ConversationMessage(role="user", content=tool_results)

# Use custom handler
agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Agent",
    description="...",
    tool_config={
        "tool": tools,
        "useToolHandler": custom_tool_handler,
        "toolMaxRecursions": 10
    }
))
```

### 3. Retrieval Augmented Generation (RAG)

**Setup:**

```python
from agent_squad.retrievers import AmazonKBRetriever

# Create retriever
retriever = AmazonKBRetriever(
    knowledge_base_id="KB123456789",
    region="us-east-1",
    max_results=5
)

# Create agent with retriever
agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Documentation Agent",
    description="Answers questions about product documentation using knowledge base",
    retriever=retriever,
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
))
```

**How RAG Works:**

```python
# User query
"How do I configure SSL certificates?"

# 1. Agent receives query
async def process_request(input_text, ...):
    # 2. Retriever fetches relevant chunks
    system_prompt = await self._prepare_system_prompt(input_text)

# _prepare_system_prompt implementation
async def _prepare_system_prompt(input_text: str) -> str:
    system_prompt = self.system_prompt

    if self.retriever:
        # Retrieve relevant chunks from knowledge base
        retrieved_context = await self.retriever.retrieve_and_combine_results(input_text)
        # retrieved_context = """
        # Chunk 1: To configure SSL certificates, navigate to Settings > Security...
        #
        # Chunk 2: SSL certificates must be in PEM format and include...
        #
        # Chunk 3: You can upload certificates using the CLI: aws configure-ssl...
        # """

        # Append to system prompt
        system_prompt += f"\n\nHere is the context to use to answer the user's question:\n{retrieved_context}"

    return system_prompt

# 3. LLM generates response with context
# System Prompt becomes:
"""
You are a Documentation Agent.
Answers questions about product documentation using knowledge base
...

Here is the context to use to answer the user's question:
Chunk 1: To configure SSL certificates, navigate to Settings > Security...

Chunk 2: SSL certificates must be in PEM format and include...

Chunk 3: You can upload certificates using the CLI: aws configure-ssl...
"""

# 4. LLM response includes information from retrieved chunks
"To configure SSL certificates, navigate to Settings > Security in the dashboard.
Your certificates must be in PEM format and include both the certificate and private key.
You can also upload certificates using the AWS CLI with the command: aws configure-ssl..."
```

### 4. Callbacks for Observability

**Complete Callback Example:**

```python
from uuid import UUID
from typing import Any, Optional

class CustomCallbacks(AgentCallbacks):

    async def on_agent_start(
        self,
        agent_name: str,
        payload_input: Any,
        messages: list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> dict:
        """Track agent start"""
        print(f"\n🤖 Agent '{agent_name}' started")
        print(f"   Input: {payload_input}")
        print(f"   History: {len(messages)} messages")

        # Return tracking info (available to all other callbacks)
        return {
            "start_time": time.time(),
            "input_length": len(str(payload_input)),
            "history_length": len(messages)
        }

    async def on_agent_end(
        self,
        agent_name: str,
        response: Any,
        messages: list[Any],
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """Track agent completion"""
        agent_tracking_info = kwargs.get('agent_tracking_info', {})
        start_time = agent_tracking_info.get('start_time', 0)
        duration = time.time() - start_time

        print(f"\n✅ Agent '{agent_name}' completed in {duration:.2f}s")

        # Log to monitoring service
        await log_to_datadog({
            "agent": agent_name,
            "duration": duration,
            "status": "success"
        })

    async def on_llm_start(
        self,
        name: str,
        payload_input: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """Track LLM invocation start"""
        print(f"   🧠 LLM call starting...")

    async def on_llm_new_token(
        self,
        token: str,
        **kwargs: Any
    ) -> None:
        """Stream tokens to UI"""
        thinking = kwargs.get('thinking', False)

        if thinking:
            # Thinking token
            print(f"[Think: {token}]", end='')
        else:
            # Regular token
            print(token, end='', flush=True)

            # Send to websocket
            await websocket.send(json.dumps({
                "type": "token",
                "content": token
            }))

    async def on_llm_end(
        self,
        name: str,
        output: Any,
        run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any
    ) -> Any:
        """Track LLM completion"""
        usage = kwargs.get('usage', {})
        input_tokens = usage.get('inputTokens', 0)
        output_tokens = usage.get('outputTokens', 0)

        print(f"\n   📊 Tokens - Input: {input_tokens}, Output: {output_tokens}")

        # Track costs
        await track_token_usage(name, input_tokens, output_tokens)

# Use callbacks
callbacks = CustomCallbacks()

agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Support Agent",
    description="Customer support",
    callbacks=callbacks,
    streaming=True
))
```

**Output:**
```
🤖 Agent 'Support Agent' started
   Input: How do I reset my password?
   History: 0 messages

   🧠 LLM call starting...
   To reset your password, follow these steps:
   1. Go to the login page
   2. Click "Forgot Password"
   3. Enter your email address
   4. Check your email for reset link

   📊 Tokens - Input: 1245, Output: 387

✅ Agent 'Support Agent' completed in 2.34s
```

### 5. Extended Thinking Mode

**Configuration:**

```python
agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Reasoning Agent",
    description="Solves complex problems with extended thinking",
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0",
    streaming=True,
    additional_model_request_fields={
        "thinking": {
            "type": "enabled"  # Enable extended thinking
        }
    }
))
```

**Usage:**

```python
response = await orchestrator.route_request(
    "Solve this complex math problem: If a train leaves Chicago at 2pm going 60mph...",
    user_id="user_123",
    session_id="session_456",
    stream_response=True
)

if response.streaming:
    thinking_content = []
    response_content = []

    async for chunk in response.output:
        if isinstance(chunk, AgentStreamResponse):
            # Thinking (reasoning process)
            if chunk.thinking:
                thinking_content.append(chunk.thinking)
                print(f"\n[Thinking: {chunk.thinking}]")

            # Regular response
            if chunk.text:
                response_content.append(chunk.text)
                print(chunk.text, end='', flush=True)

            # Final message with complete thinking
            if chunk.final_message:
                complete_thinking = chunk.final_thinking
                print(f"\n\nComplete reasoning:\n{complete_thinking}")
```

**Example Output:**
```
[Thinking: Let me break down this problem step by step...]
[Thinking: First, I need to determine the distance between cities...]
[Thinking: Given the speed of 60mph and departure time of 2pm...]
[Thinking: I should calculate: distance = speed × time...]

The train will arrive at approximately 5:30pm.

Complete reasoning:
Let me break down this problem step by step. First, I need to determine
the distance between cities, which is 210 miles based on the problem.
Given the speed of 60mph and departure time of 2pm, I should calculate:
distance = speed × time, so time = distance / speed = 210 / 60 = 3.5 hours.
Adding 3.5 hours to 2pm gives us 5:30pm.
```

---

## Example Use Cases

### 1. Customer Support System

```python
from agent_squad import AgentSquad
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions, LexBotAgent
from agent_squad.classifiers import BedrockClassifier
from agent_squad.storage import DynamoDBChatStorage

# Setup storage
storage = DynamoDBChatStorage(
    table_name="customer-support-conversations",
    region="us-east-1"
)

# Setup classifier
classifier = BedrockClassifier()

# Create orchestrator
orchestrator = AgentSquad(
    storage=storage,
    classifier=classifier
)

# Billing Agent
billing_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Billing Agent",
    description="Handles billing inquiries, payment issues, refunds, and subscription management",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    streaming=True
))

# Technical Support Agent
tech_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Technical Support",
    description="Troubleshoots technical issues, helps with product setup, and resolves bugs",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    streaming=True
))

# Order Tracking Agent (Lex Bot)
order_agent = LexBotAgent({
    "name": "Order Tracking",
    "description": "Tracks order status, shipping updates, and delivery information",
    "botId": os.environ['LEX_BOT_ID'],
    "botAliasId": os.environ['LEX_BOT_ALIAS_ID'],
    "localeId": "en_US"
})

# Returns & Exchanges Agent
returns_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Returns Agent",
    description="Processes returns, exchanges, and handles damaged product claims",
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
))

# Register agents
orchestrator.add_agent(billing_agent)
orchestrator.add_agent(tech_agent)
orchestrator.add_agent(order_agent)
orchestrator.add_agent(returns_agent)

# Set default agent for unclear queries
default_agent = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="General Support",
    description="Handles general inquiries and routes to appropriate department",
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
))
orchestrator.set_default_agent(default_agent)

# Handle customer request
async def handle_customer_request(customer_id: str, message: str, session_id: str):
    response = await orchestrator.route_request(
        user_input=message,
        user_id=customer_id,
        session_id=session_id,
        stream_response=True
    )

    if response.streaming:
        async for chunk in response.output:
            yield chunk.text
    else:
        yield response.output.content[0]['text']

# Example usage
async for response_chunk in handle_customer_request(
    customer_id="CUST_12345",
    message="I was charged twice for my last order",
    session_id="SESSION_67890"
):
    print(response_chunk, end='', flush=True)
```

### 2. AI Production Studio (SupervisorAgent)

```python
from agent_squad import AgentSquad
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions, SupervisorAgent, SupervisorAgentOptions
from agent_squad.storage import InMemoryChatStorage

# Create specialized team members
director = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Film Director",
    description="Creative director responsible for overall vision, storytelling, and artistic decisions",
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
))

screenwriter = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Screenwriter",
    description="Writes scripts, dialogue, and narrative structure for productions",
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
))

cinematographer = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Cinematographer",
    description="Plans camera angles, lighting, shot composition, and visual aesthetics",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
))

editor = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Video Editor",
    description="Handles post-production, cutting, transitions, pacing, and final assembly",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
))

sound_designer = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Sound Designer",
    description="Creates audio landscape, music selection, sound effects, and audio mixing",
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
))

# Create supervisor to coordinate the team
production_coordinator = SupervisorAgent(SupervisorAgentOptions(
    name="Production Coordinator",
    description="Coordinates the entire film production team to create commercials, short films, and video content",
    lead_agent=director,  # Director leads the coordination
    team=[screenwriter, cinematographer, editor, sound_designer],
    storage=InMemoryChatStorage(),
    trace=True  # Enable logging to see agent interactions
))

# Create main orchestrator
orchestrator = AgentSquad()
orchestrator.add_agent(production_coordinator)

# Production request
production_brief = """
Create a 30-second commercial for "EcoWalk" - sustainable, eco-friendly running shoes.
Target audience: 25-40 year old environmentally conscious fitness enthusiasts.
Key message: Performance meets sustainability.
Tone: Inspiring, energetic, authentic.
"""

response = await orchestrator.route_request(
    user_input=production_brief,
    user_id="client_123",
    session_id="project_ecowalk_001"
)

print(response.output.content[0]['text'])
```

**What Happens Internally:**

```
1. Production Coordinator (Director) receives brief

2. Director uses send_messages tool:
   {
     "messages": [
       {
         "recipient": "Screenwriter",
         "content": "Write a 30-second commercial script for EcoWalk eco-friendly running shoes. Target: 25-40 fitness enthusiasts. Message: Performance meets sustainability. Tone: Inspiring, energetic."
       },
       {
         "recipient": "Cinematographer",
         "content": "Plan visuals for 30-second EcoWalk shoe commercial. Show: runner in nature, shoe details, sustainability aspect. Style: Clean, energetic, natural lighting."
       },
       {
         "recipient": "Sound Designer",
         "content": "Audio concept for EcoWalk commercial: uplifting music, natural sounds, energetic vibe for 30 seconds."
       }
     ]
   }

3. Agents work in parallel:
   - Screenwriter creates script with voiceover and scenes
   - Cinematographer plans 6 key shots with camera movements
   - Sound Designer selects music and sound effects

4. Director receives all responses and uses send_messages again:
   {
     "messages": [
       {
         "recipient": "Video Editor",
         "content": "Based on script and shot list, create editing plan: pacing, transitions, 30-second assembly."
       }
     ]
   }

5. Editor creates post-production plan

6. Director synthesizes everything into final production package:
   - Script
   - Shot list
   - Lighting plan
   - Audio design
   - Editing timeline
   - Final deliverables
```

### 3. Healthcare Coordination System

```python
# Medical team
primary_doctor = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Primary Care Physician",
    description="Diagnoses conditions, prescribes treatments, coordinates care",
    model_id="anthropic.claude-3-5-sonnet-20240620-v1:0"
))

pharmacist = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Pharmacist",
    description="Reviews medications, checks interactions, provides dosage information",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
))

insurance_specialist = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Insurance Specialist",
    description="Verifies coverage, explains benefits, processes claims",
    model_id="anthropic.claude-3-sonnet-20240229-v1:0"
))

lab_technician = BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Lab Technician",
    description="Explains lab tests, results interpretation, scheduling",
    model_id="anthropic.claude-3-haiku-20240307-v1:0"
))

# Coordinator
care_coordinator = SupervisorAgent(SupervisorAgentOptions(
    name="Care Coordinator",
    description="Coordinates patient care across medical team members",
    lead_agent=primary_doctor,
    team=[pharmacist, insurance_specialist, lab_technician],
    storage=InMemoryChatStorage()
))

orchestrator = AgentSquad()
orchestrator.add_agent(care_coordinator)

# Patient inquiry
response = await orchestrator.route_request(
    user_input="I need to start blood pressure medication. What do I need to know about costs, side effects, and testing?",
    user_id="patient_456",
    session_id="visit_789"
)
```

**Coordinator Response Includes:**
- Doctor: Medication recommendation and monitoring plan
- Pharmacist: Side effects, interactions, how to take medication
- Insurance: Coverage details, copay information
- Lab: Required blood tests, scheduling, what to expect

---

## Configuration & Deployment

### Python Installation

```bash
# Core with all providers
pip install agent-squad[all]

# AWS services only
pip install agent-squad[aws]

# Anthropic only
pip install agent-squad[anthropic]

# OpenAI only
pip install agent-squad[openai]

# Minimal (no LLM providers, bring your own)
pip install agent-squad
```

### TypeScript Installation

```bash
npm install agent-squad
```

### Environment Variables

```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# Amazon Lex
LEX_BOT_ID=your_bot_id
LEX_BOT_ALIAS_ID=your_bot_alias_id

# Anthropic
ANTHROPIC_API_KEY=your_api_key

# OpenAI
OPENAI_API_KEY=your_api_key

# DynamoDB Storage
DYNAMODB_TABLE_NAME=conversations
DYNAMODB_REGION=us-east-1

# Knowledge Bases
BEDROCK_KB_ID=your_knowledge_base_id
```

### AWS Lambda Deployment

**Python Lambda:**

```python
# lambda_handler.py
import json
from agent_squad import AgentSquad
from agent_squad.agents import BedrockLLMAgent, BedrockLLMAgentOptions
from agent_squad.storage import DynamoDBChatStorage
from agent_squad.classifiers import BedrockClassifier

# Initialize outside handler for warm starts
storage = DynamoDBChatStorage(
    table_name="conversations",
    region="us-east-1"
)

classifier = BedrockClassifier()
orchestrator = AgentSquad(storage=storage, classifier=classifier)

# Register agents
orchestrator.add_agent(BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Tech Support",
    description="Technical support agent"
)))

orchestrator.add_agent(BedrockLLMAgent(BedrockLLMAgentOptions(
    name="Billing Agent",
    description="Billing and payments"
)))

def lambda_handler(event, context):
    body = json.loads(event['body'])

    response = await orchestrator.route_request(
        user_input=body['message'],
        user_id=body['userId'],
        session_id=body['sessionId']
    )

    return {
        'statusCode': 200,
        'body': json.dumps({
            'response': response.output.content[0]['text'],
            'agent': response.metadata.agent_name
        })
    }
```

**requirements.txt:**
```
agent-squad[aws]==1.0.0
```

**Lambda Configuration:**
- Runtime: Python 3.11
- Memory: 512 MB
- Timeout: 30 seconds
- Environment variables: AWS_REGION, DYNAMODB_TABLE_NAME
- IAM Role: Bedrock invoke, DynamoDB read/write

---

## Code Quality & Best Practices

### Architecture Strengths

1. **Clean Separation of Concerns:**
   - Orchestrator: Coordination only
   - Agents: Processing logic
   - Classifier: Intent detection
   - Storage: Persistence
   - No circular dependencies

2. **Extensibility:**
   - Abstract base classes for all major components
   - Easy to add custom agents, classifiers, storage
   - Plugin architecture

3. **Type Safety:**
   - Strong typing with dataclasses
   - Type hints throughout
   - Enums for constants

4. **Async/Await:**
   - Fully async architecture
   - Non-blocking I/O
   - Efficient resource usage

5. **Error Handling:**
   - Try/catch blocks around LLM calls
   - Graceful degradation
   - Logging of errors

### Design Patterns Used

1. **Strategy Pattern:** Different classifier and storage implementations
2. **Factory Pattern:** Agent creation
3. **Observer Pattern:** Callbacks for agent lifecycle events
4. **Template Method:** Base agent class with hooks
5. **Composite Pattern:** SupervisorAgent manages team of agents

### Best Practices Observed

1. **DRY (Don't Repeat Yourself):**
   - Shared utilities (conversation_to_dict, format_messages)
   - Base classes with common functionality

2. **Single Responsibility:**
   - Each class has one clear purpose
   - Orchestrator doesn't implement LLM logic
   - Agents don't handle storage

3. **Open/Closed Principle:**
   - Easy to extend (add new agents)
   - No need to modify core orchestrator

4. **Dependency Injection:**
   - Storage, classifier, agents injected
   - Easy testing and swapping implementations

5. **Configuration Management:**
   - Centralized config in AgentSquadConfig
   - Environment-based settings
   - Sensible defaults

### Testing Structure

```
tests/
├── agents/
│   ├── test_agent.py
│   ├── test_bedrock_llm_agent.py
│   ├── test_supervisor_agent.py
│   └── ...
├── classifiers/
│   ├── test_classifier.py
│   ├── test_bedrock_classifier.py
│   └── ...
├── storage/
│   ├── test_chat_storage.py
│   ├── test_in_memory_chat_storage.py
│   ├── test_dynamodb_chat_storage.py
│   └── ...
└── test_orchestrator.py
```

---

## Conclusion

The Agent Squad framework is a well-architected, production-ready solution for building multi-agent conversational AI systems. Its strengths include:

**Technical Excellence:**
- Clean modular architecture
- Strong typing and type safety
- Comprehensive async support
- Extensive error handling
- Good test coverage

**Developer Experience:**
- Dual language support (Python & TypeScript)
- Consistent APIs across languages
- Rich documentation and examples
- Easy to extend and customize
- Sensible defaults

**Production Readiness:**
- Multiple storage backends
- Scalable architecture
- AWS Lambda support
- Monitoring via callbacks
- Cost tracking capabilities

**Advanced Capabilities:**
- SupervisorAgent for team coordination
- Streaming responses
- Tool/function calling
- RAG integration
- Extended thinking mode
- Parallel agent processing

The framework successfully abstracts the complexity of managing multiple AI agents while providing flexibility for customization and extension. It's suitable for a wide range of applications from simple chatbots to complex multi-agent systems with hierarchical coordination.
