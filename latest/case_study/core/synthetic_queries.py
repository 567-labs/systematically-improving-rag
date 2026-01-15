"""
Synthetic query generation for RAG evaluation.

Implements Workshop Chapter 1 concepts:
- v1: Content-focused queries (what users asked about) → 62% recall
- v2: Pattern-focused queries (how users phrased it) → 11% recall on first messages

The Alignment Problem: v1 queries work well with first-message embeddings because 
they're both content-focused. v2 queries need different embeddings (summaries) because 
they focus on conversational patterns. This 50% performance gap demonstrates why 
matching query strategy to embedding strategy matters more than reranking.

See Workshop Chapter 5.2 for how this parallels blueprint search: task-specific 
summaries (85%) vs generic embeddings (27%).
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class SearchQueries(BaseModel):
    """Generated search queries that could lead to discovering a conversation."""

    chain_of_thought: str = Field(
        description="Chain of thought process for generating the search queries"
    )
    queries: List[str] = Field(
        description="4-7 diverse search queries that users might type to find this conversation",
        min_items=3,
        max_items=8,
    )


async def synthetic_question_generation_v1(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> SearchQueries:
    """
    Generate diverse synthetic search queries from a chat conversation.

    As a product manager analyzing ChatGPT usage patterns, this function creates
    search queries that users might have typed to discover similar conversations.
    The queries should be diverse and cover different aspects of the conversation.

    Args:
        client: instructor-patched client
        conversation: Dictionary containing conversation data with 'messages' or 'conversation' key

    Returns:
        SearchQueries object with 4-5 diverse search queries and reasoning
    """

    prompt = """
    You are a product manager analyzing ChatGPT usage patterns. Your goal is to understand 
    how users might search to find conversations like this one.
    
    Given this conversation, generate 4-5 diverse search queries that different users might 
    type when looking for similar help or information. The queries should:
    
    1. Cover different aspects of the conversation (technical terms, problem description, solution type)
    2. Vary in specificity (some broad, some specific)
    3. Use different phrasings and vocabulary levels
    4. Reflect natural user search behavior
    5. Include both question-style and keyword-style queries
    
    <conversation>
    {% for message in messages %}
        <message role="{{ message.role }}">
            {{ message.content }}
        </message>
    {% endfor %}
    </conversation>
    
    Generate queries that would realistically lead someone to discover this conversation.
    """

    response = await client.chat.completions.create(
        response_model=SearchQueries,
        messages=[{"role": "user", "content": prompt}],
        context={"messages": messages},
    )

    return response


async def synthetic_question_generation_v2(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> SearchQueries:
    """
    Generate search queries for finding conversations with similar patterns and characteristics.

    This version focuses on identifying conversation types, themes, and patterns that would be
    useful for researchers, content moderators, or analysts studying human-AI interactions.

    Args:
        client: instructor-patched client
        conversation_id: ID of the conversation to analyze
        conversation: Dictionary containing conversation data with 'messages' or 'conversation' key
        conversation_hash: Unique hash of the conversation for caching

    Returns:
        SearchQueries object with pattern-focused search queries
    """

    prompt = """
    You are a research analyst studying patterns in human-AI conversations from the WildChat dataset.
    Your goal is to identify the key characteristics and patterns in this conversation that would help
    researchers find similar types of conversations.
    
    Analyze this conversation and generate search queries that would help find conversations with:
    - Similar content themes or domains (medical, creative, technical, etc.)
    - Similar user intents (seeking advice, creative collaboration, testing AI limits, etc.)
    - Similar interaction patterns (role-playing, Q&A, refusal situations, etc.)
    - Similar AI behaviors or response types
    
    Focus on generating queries that capture the ESSENCE and PATTERNS rather than specific details.
    
    Examples of good pattern queries:
    - "conversations where users ask about medical diagnoses"
    - "role-playing scenarios with fictional characters"
    - "conversations where AI refuses medical advice"
    - "creative writing collaborations"
    - "technical troubleshooting discussions"
    - "conversations testing AI content policies"
    - "users seeking relationship advice"
    - "educational Q&A about scientific concepts"
    
    <conversation>
    {% for message in messages %}
        <message role="{{ message.role }}">
            {{ message.content }}
        </message>
    {% endfor %}
    </conversation>
    
    Generate 5-7 search queries that focus on conversation patterns, themes, and characteristics
    rather than specific content details. Think about what makes this conversation type distinct
    and how researchers would categorize it.
    """

    response = await client.chat.completions.create(
        response_model=SearchQueries,
        messages=[
            {
                "role": "system",
                "content": "You are an expert conversation analyst specializing in categorizing and understanding patterns in human-AI interactions. Focus on identifying conversation types, themes, and structural patterns rather than specific content details.",
            },
            {"role": "user", "content": prompt},
        ],
        context={"messages": messages},
    )

    return response


async def synthetic_question_generation_v3(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> SearchQueries:
    """
    Generate highly specific search queries that balance pattern recognition with unique details.

    This version creates queries that:
    - Maintain awareness of conversation patterns and types
    - Include specific distinguishing details
    - Capture customer satisfaction/frustration signals
    - Are specific enough to find individual conversations

    Args:
        client: instructor-patched client
        messages: List of conversation messages

    Returns:
        SearchQueries object with specific, pattern-aware search queries
    """

    prompt = """
    You are a search optimization specialist analyzing conversations from the WildChat dataset.
    Your goal is to create search queries that would uniquely identify THIS SPECIFIC conversation
    while maintaining awareness of its patterns and characteristics.
    
    Analyze this conversation and generate 5-7 search queries that:
    
    1. COMBINE pattern descriptions with specific details:
       - Start with the conversation type/pattern
       - Add unique technical terms, concepts, or specifics mentioned
       - Include memorable phrases, examples, or distinctive elements
    
    2. CAPTURE user satisfaction signals:
       - User frustration: "conversation where user gets frustrated about [specific issue]"
       - User satisfaction: "helpful AI conversation successfully solving [specific problem]"
       - Failed attempts: "AI struggling to understand user's request about [topic]"
       - Misunderstandings: "conversation with miscommunication about [specific concept]"
    
    3. INCLUDE distinguishing combinations:
       - Multiple topics discussed together
       - Specific technical stack or tools mentioned
       - Unique examples or scenarios presented
       - Particular error messages or issues
    
    4. USE this query structure:
       - "[conversation type] + [specific topic/issue] + [distinguishing detail]"
       - Examples:
         * "technical troubleshooting Docker PostgreSQL connection refused error"
         * "frustrated user requesting medical advice AI refuses to provide"
         * "role-playing conversation medieval blacksmith discussing enchanted sword crafting"
         * "AI successfully helping debug React useState infinite loop issue"
         * "creative writing collaboration vampire romance story plot twist"
    
    5. IDENTIFY unique aspects that distinguish this from similar conversations:
       - Specific error messages or technical details
       - Unusual combinations of topics
       - Particular user reactions or feedback
       - Distinctive AI responses or behaviors
    
    IMPORTANT: Each query should be specific enough that someone could find THIS conversation
    among thousands of similar ones, while still being natural search queries.
    
    <conversation>
    {% for message in messages %}
        <message role="{{ message.role }}">
            {{ message.content }}
        </message>
    {% endfor %}
    </conversation>
    
    Generate queries that uniquely identify this conversation while capturing its patterns,
    user satisfaction level, and distinguishing features.
    """

    response = await client.chat.completions.create(
        response_model=SearchQueries,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating specific, distinguishing search queries that balance pattern recognition with unique details. Focus on what makes each conversation unique while maintaining awareness of its type and user satisfaction signals.",
            },
            {"role": "user", "content": prompt},
        ],
        context={"messages": messages},
    )

    return response


async def synthetic_question_generation_v5(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> SearchQueries:
    """Generate queries optimized for AI agent analysis of system failures and improvements."""

    prompt = """
    Analyze this conversation and generate 5-7 search queries that an AI analysis agent would use 
    to find similar patterns for system improvement. Your queries should help identify:
    
    1. ROOT CAUSE PATTERNS:
       - "AI failed to [specific capability] when user [specific context]"
       - "misunderstanding about [concept] leading to [consequence]"
       - "system limitation in [feature] causing [user reaction]"
    
    2. FAILURE MODE QUERIES:
       - "conversation where AI [specific failure] despite [attempts to help]"
       - "[error type] occurring when [specific conditions]"
       - "repeated clarification needed for [topic] due to [root cause]"
    
    3. RECOVERY AND RESOLUTION:
       - "AI successfully recovered from [initial failure] by [specific action]"
       - "user satisfaction improved after [specific intervention]"
       - "workaround provided for [limitation] using [alternative approach]"
    
    4. IMPACT AND SEVERITY:
       - "high-impact failure in [domain] affecting [user goal]"
       - "critical misunderstanding about [topic] preventing [outcome]"
       - "user abandoned task due to [specific issue]"
    
    5. IMPROVEMENT OPPORTUNITIES:
       - "conversation revealing need for [specific capability]"
       - "user requesting [feature] not currently supported"
       - "pattern of confusion about [concept] suggesting [improvement]"
    
    Structure each query to include:
    - The failure pattern or success pattern
    - Specific technical details or domain
    - Root cause indicators
    - User impact or satisfaction level
    - Potential improvement direction
    
    <conversation>
    {% for message in messages %}
        <message role="{{ message.role }}">
            {{ message.content }}
        </message>
    {% endfor %}
    </conversation>
    
    Generate queries that would help an AI agent identify systemic issues and improvement opportunities.
    """

    response = await client.chat.completions.create(
        response_model=SearchQueries,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating analytical queries for AI system improvement. Focus on identifying failure patterns, root causes, and improvement opportunities that would help an AI agent make data-driven recommendations.",
            },
            {"role": "user", "content": prompt},
        ],
        context={"messages": messages},
    )

    return response
