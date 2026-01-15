"""
Conversation summarization strategies for improved retrieval.

Implements Workshop Chapter 5.2 concepts - synthetic text generation as compression:
- v1: Search-optimized (content focus) → works well with v1 queries
- v3: Balanced approach
- v4: Pattern-optimized (conversation style) → works well with v2 queries  
- v5: Hybrid (best of both) → 82.0% v1 recall, 55.0% v2 recall

The Solution to Alignment Problem: When queries don't match embeddings, change the 
embeddings! Summaries bridge the gap between what users search for and how conversations 
are structured.

Parallels Workshop Chapter 5.2 blueprint search: vision-generated summaries describing 
spatial features (room counts, dimensions) instead of raw image embeddings. Same concept: 
create searchable text that matches user mental models.
"""

from typing import List, Dict, Any
from pydantic import BaseModel, Field


class ConversationSummary(BaseModel):
    """Generated summary of a conversation."""

    chain_of_thought: str = Field(
        description="Chain of thought process for generating the summary"
    )
    summary: str = Field(
        description="The conversation summary based on the specified approach"
    )


async def conversation_summary_v1(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> ConversationSummary:
    """
    Generate a concise summary focused on key topics and searchable terms.

    This approach creates summaries optimized for semantic search by focusing on:
    - Main topics discussed
    - Key questions asked
    - Primary solutions or information provided
    - Important technical terms or concepts

    The summary should be 2-3 sentences that capture the essence of what a user
    might search for to find this conversation.

    Args:
        client: instructor-patched client
        messages: List of conversation messages

    Returns:
        ConversationSummary object with concise, search-optimized summary
    """

    prompt = """
    You are creating search-optimized summaries of conversations for a RAG system.
    Your goal is to summarize this conversation in a way that makes it easily discoverable
    through semantic search.
    
    Generate a 2-3 sentence summary that:
    1. Captures the main topic or problem discussed
    2. Includes key technical terms, concepts, or domain-specific vocabulary
    3. Mentions the type of assistance provided (explanation, troubleshooting, code example, etc.)
    4. Uses natural language that someone might use when searching
    
    Focus on WHAT the conversation is about, not HOW it progresses.
    
    <conversation>
    {% for message in messages %}
        <message role="{{ message.role }}">
            {{ message.content }}
        </message>
    {% endfor %}
    </conversation>
    
    Generate a concise, searchable summary of this conversation.
    """

    response = await client.chat.completions.create(
        response_model=ConversationSummary,
        messages=[{"role": "user", "content": prompt}],
        context={"messages": messages},
        max_retries=3,
    )

    return response


async def conversation_summary_v2(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> ConversationSummary:
    """
    Generate a comprehensive summary that captures conversation patterns, themes, and user interaction dynamics.

    This approach creates summaries that capture:
    - Conversation patterns and interaction types
    - User intent, behavior, and preferences
    - AI response characteristics and adjustments
    - Overall conversation flow and structure
    - User feedback and emotional context
    - Knowledge requests and learning progression
    - User Satisfaction and Frustration Points

    The summary should be detailed enough to match pattern-based queries and understand
    the full context of user-AI interaction dynamics.

    Args:
        client: instructor-patched client
        messages: List of conversation messages

    Returns:
        ConversationSummary object with comprehensive, pattern-focused summary
    """

    prompt = """
Your task is to create a detailed summary of the conversation, paying close attention to the user's explicit requests, feedback, and the assistant's responses.
This summary should be thorough in capturing key topics, important details, interaction patterns, and user preferences that would be essential for understanding the conversation's context and continuation.

Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points. In your analysis process:

1. Chronologically analyze each message and section of the conversation. For each section thoroughly identify:
   - The user's explicit requests and intents
   - The assistant's approach to addressing the user's requests
   - Key decisions, important concepts, and significant details
   - Specific information, examples, or resources mentioned
   - User feedback, preferences, and emotional responses
   - Learning progression and context switches
2. Double-check for accuracy and completeness, addressing each required element thoroughly.

Your summary should include the following sections:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail

2. Knowledge Requests: Identify specific knowledge, explanations, or information the user is seeking from the AI

3. User Feedback and Preferences: Document any feedback about the AI's responses, communication style preferences, or corrections requested by the user (e.g., "be less verbose," "provide more examples," "explain differently")

4. Key Topics and Concepts: List all important topics, concepts, and subject matter discussed

5. Information and Resources: Enumerate specific information, examples, resources, or references provided or discussed. Pay special attention to the most recent messages and include important details where applicable

6. Problem Solving: Document problems addressed and any ongoing efforts to resolve issues

7. Conversation Dynamics: Note the user's communication style, level of expertise, emotional state (frustration, satisfaction, confusion), and any adjustments made to match their needs

8. User Frustration Points: Specifically identify and quote instances where the user expresses significant frustration, impatience, or dissatisfaction with the AI's responses or the conversation flow

9. Learning Progression: Track how the user's understanding evolves throughout the conversation and any knowledge building that occurs

10. Context Switches: Identify when the conversation shifts topics or focus areas, and note any repeated patterns or recurring themes

11. Clarifications and Corrections: Track instances where the user needed to clarify their request or correct the AI's understanding

12. User Goals and Success Metrics: Distinguish between immediate requests and broader objectives, noting what constitutes success for the user

13. Assumptions Made: Track assumptions the AI made about user needs, context, or expertise level

14. Pending Tasks: Outline any pending tasks or follow-up items that have been explicitly requested

15. Current Focus: Describe in detail precisely what was being discussed or worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant

16. Optional Next Step: List the next step that would logically follow based on the most recent work or discussion. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests and the topic being addressed immediately before this summary request. If the last topic was concluded, then only list next steps if they are explicitly in line with the user's request

17. If there is a next step, include direct quotes from the most recent conversation showing exactly what was being discussed and where the conversation left off. This should be verbatim to ensure there's no drift in topic interpretation

18. User Journey & Task Completion: Analyze the user's workflow - did they complete their intended task? Where did they get stuck? How many exchanges were needed? What was their entry point and approach?

19. AI Performance Issues: Identify instances where the AI failed to understand, provided incorrect information, or hit capability limits. Note any hallucinations or clear mistakes.

20. User Behavior Patterns: Document the user's information-seeking strategies, trust indicators (verification, doubt), and usage intensity (quick question vs. deep exploration).

21. Product Experience Insights: Note any feature discoveries, expectation mismatches, or workaround behaviors the user developed to handle AI limitations.

22. Conversation Quality Indicators: Track satisfaction signals (thanks, success expressions), abandonment patterns, and any repetitive questioning that suggests the user wasn't getting what they needed.

Here's an example of how your output should be structured:

<example>
<chain_of_thought>
[Your thought process, ensuring all points are covered thoroughly and accurately]
</chain_of_thought>

<summary>
1. Primary Request and Intent:
   [Detailed description]

2. Knowledge Requests:
   - [Specific knowledge/explanation sought 1]
   - [Specific knowledge/explanation sought 2]
   - [...]

3. User Feedback and Preferences:
   - [Feedback about AI responses, style preferences, corrections]
   - [Communication preferences expressed]
   - [...]

4. Key Topics and Concepts:
   - [Topic/Concept 1]
   - [Topic/Concept 2]
   - [...]

5. Information and Resources:
   - [Resource/Information 1]
      - [Summary of why this information is important]
      - [Summary of how it was used or discussed]
      - [Important details or examples]
   - [...]

6. Problem Solving:
   [Description of problems addressed and ongoing efforts]

7. Conversation Dynamics:
   - User expertise level: [beginner/intermediate/advanced]
   - Communication style: [direct/detailed/collaborative/etc.]
   - Emotional state: [engaged/frustrated/satisfied/confused/etc.]
   - Adjustments made: [any changes to approach based on user feedback]

8. User Frustration Points:
   - [Direct quotes showing user frustration, impatience, or dissatisfaction]
   - [Context around what triggered the frustration]
   - [How the AI responded to user frustration]

9. Learning Progression:
   - [How user's understanding evolved]
   - [Knowledge building that occurred]
   - [Skills or concepts mastered]

10. Context Switches:
    - [Topic shifts and focus changes]
    - [Repeated patterns or recurring themes]

11. Clarifications and Corrections:
    - [Instances where user clarified or corrected]
    - [Misunderstandings that were resolved]

12. User Goals and Success Metrics:
    - Immediate goals: [short-term objectives]
    - Broader objectives: [long-term aims]
    - Success indicators: [what constitutes success]

13. Assumptions Made:
    - [Assumptions about user needs or context]
    - [Assumptions about expertise level]
    - [Assumptions about preferred approach]

14. Pending Tasks:
    - [Task 1]
    - [Task 2]
    - [...]

15. Current Focus:
    [Precise description of current discussion or work]

16. Optional Next Step:
    [Optional next step to take]

17. User Journey & Task Completion:
    - Task completion status: [completed/partially completed/abandoned]
    - Workflow efficiency: [number of exchanges needed, bottlenecks]
    - Entry point: [how user started the conversation]
    - Approach strategy: [user's problem-solving approach]

18. AI Performance Issues:
    - Failure points: [where AI failed to understand or help]
    - Incorrect information: [any hallucinations or mistakes]
    - Capability gaps: [tasks user wanted that AI couldn't handle]

19. User Behavior Patterns:
    - Information seeking: [how user phrases questions, iteration patterns]
    - Trust indicators: [verification behaviors, expressions of doubt]
    - Usage intensity: [quick question vs. deep exploration]

20. Product Experience Insights:
    - Feature discoveries: [new capabilities user found]
    - Expectation mismatches: [where user expected different behavior]
    - Workarounds: [strategies user developed for AI limitations]

21. Conversation Quality Indicators:
    - Satisfaction signals: [thanks, success expressions, positive feedback]
    - Abandonment patterns: [abrupt endings, unresolved issues]
    - Repetitive patterns: [similar questions asked multiple times]

</summary>
</example>

Please provide your summary based on the conversation so far, following this structure and ensuring precision and thoroughness in your response.

{% for message in messages %}
    <message role="{{ message.role }}">
        {{ message.content }}
    </message>
{% endfor %}
    """

    response = await client.chat.completions.create(
        response_model=ConversationSummary,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing and categorizing human-AI conversations, focusing on patterns, themes, interaction characteristics, and user experience dynamics across all types of conversations.",
            },
            {"role": "user", "content": prompt},
        ],
        context={"messages": messages},
        max_retries=3,
    )

    return response


async def conversation_summary_v3(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> ConversationSummary:
    """
    Generate a concise pattern-focused summary that bridges content and patterns.

    This approach creates 3-5 sentence summaries that capture:
    - Conversation type and category
    - Main interaction pattern
    - Key topics with pattern context
    - User intent category
    - Notable characteristics

    The summary balances brevity with pattern recognition, making it useful
    for both content and pattern-based searches.

    Args:
        client: instructor-patched client
        messages: List of conversation messages

    Returns:
        ConversationSummary object with concise pattern-aware summary
    """

    prompt = """
You are creating balanced summaries that capture both content and conversation patterns.

Generate a 3-5 sentence summary that includes:

1. **Opening Pattern Statement**: "This is a [TYPE] conversation where [PATTERN]."
   Examples:
   - "This is a technical troubleshooting conversation where the user seeks help with database errors."
   - "This is an educational Q&A conversation where the user explores quantum physics concepts."
   - "This is a creative collaboration where the user and AI co-write a fantasy story."

2. **Interaction Dynamic**: One sentence about HOW the conversation unfolds.
   Examples:
   - "The user provides error messages and the AI diagnoses issues step-by-step."
   - "Through iterative questions, the user deepens their understanding of complex topics."
   - "The conversation follows a pattern of proposal, feedback, and refinement."

3. **Key Content**: One sentence listing the main topics/concepts discussed.
   Examples:
   - "Key topics include PostgreSQL configuration, connection pooling, and query optimization."
   - "The discussion covers wave-particle duality, quantum entanglement, and measurement."
   - "Central elements include character backstories, world-building, and plot development."

4. **User Characteristic** (optional): If notable, one sentence about the user's approach or state.
   Examples:
   - "The user demonstrates intermediate technical knowledge and methodical debugging."
   - "The user shows curiosity and asks progressively deeper questions."
   - "The user expresses frustration with initial attempts but persists toward resolution."

Focus on creating a summary that works for BOTH content searches AND pattern searches.

<conversation>
{% for message in messages %}
    <message role="{{ message.role }}">
        {{ message.content }}
    </message>
{% endfor %}
</conversation>

Generate a concise, pattern-aware summary that balances content and conversation dynamics.
    """

    response = await client.chat.completions.create(
        response_model=ConversationSummary,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at creating concise summaries that capture both what was discussed and how the conversation unfolded. Balance pattern recognition with content summary.",
            },
            {"role": "user", "content": prompt},
        ],
        context={"messages": messages},
        max_retries=3,
    )

    return response


async def conversation_summary_v4(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> ConversationSummary:
    """
    Generate a pattern-focused summary optimized for v2-style queries.

    This approach creates summaries that explicitly capture:
    - Conversation categorization and type
    - Interaction patterns and dynamics
    - Domain and thematic classification
    - User behavior patterns
    - AI response patterns
    - Meta-conversation characteristics

    The summary uses pattern-descriptive language that aligns with v2 query style.

    Args:
        client: instructor-patched client
        messages: List of conversation messages

    Returns:
        ConversationSummary object with pattern-optimized summary
    """

    prompt = """
You are a conversation analyst creating searchable summaries that capture conversation PATTERNS and TYPES, not just content details.

Your goal is to create a summary that would match pattern-based search queries like:
- "conversations involving role-playing with fictional characters"
- "technical troubleshooting discussions"
- "conversations where AI refuses medical advice"
- "educational Q&A about scientific concepts"

Analyze this conversation and create a structured summary with these components:

1. **Conversation Classification** (1-2 sentences):
   Start with: "This is a [TYPE] conversation involving/where/about..."
   Examples:
   - "This is a technical troubleshooting conversation involving database configuration issues"
   - "This is a role-playing conversation where the user asks the AI to act as a fictional character"
   - "This is an educational Q&A conversation about quantum physics concepts"

2. **Interaction Patterns** (2-3 patterns):
   Use pattern-descriptive phrases like:
   - "User seeks step-by-step guidance for [topic]"
   - "AI provides explanatory responses with examples"
   - "Conversation follows problem-diagnosis-solution pattern"
   - "User tests AI boundaries with controversial requests"
   - "Collaborative creative writing with iterative refinement"

3. **Domain and Theme Tags** (3-5 tags):
   Include categorical descriptors:
   - Technical domains: "software development", "database management", "API design"
   - Creative domains: "creative writing", "storytelling", "character development"
   - Educational domains: "science education", "language learning", "skill tutorials"
   - Interaction types: "troubleshooting", "advisory", "collaborative", "exploratory"

4. **User Behavior Patterns**:
   - "User demonstrates [expertise level] in [domain]"
   - "User exhibits [frustration/satisfaction/curiosity] about [aspect]"
   - "User's approach is [methodical/exploratory/direct/iterative]"

5. **AI Response Characteristics**:
   - "AI provides [detailed/concise/cautious] responses"
   - "AI [accepts/refuses/redirects] user requests about [topic]"
   - "AI adjusts communication style to match user's [expertise/tone/needs]"

6. **Key Content Elements** (brief):
   List the main topics/concepts discussed, but keep it concise and pattern-focused.

Remember: Focus on HOW the conversation unfolds and WHAT TYPE it represents, not just WHAT was discussed.

<conversation>
{% for message in messages %}
    <message role="{{ message.role }}">
        {{ message.content }}
    </message>
{% endfor %}
</conversation>

Generate a pattern-focused summary that would be discoverable by researchers looking for specific conversation types and interaction patterns.
"""

    response = await client.chat.completions.create(
        response_model=ConversationSummary,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at categorizing conversations by their patterns, types, and interaction dynamics. Focus on creating summaries that match how researchers search for conversation patterns rather than specific content.",
            },
            {"role": "user", "content": prompt},
        ],
        context={"messages": messages},
        max_retries=3,
    )

    return response


async def conversation_summary_v5(
    client,  # instructor-patched client
    messages: List[Dict[str, Any]],
) -> ConversationSummary:
    """
    Generate summaries optimized for AI agent failure analysis and improvement identification.

    This approach creates summaries that capture:
    - Root causes of AI failures and misunderstandings
    - User impact and satisfaction metrics
    - Recovery patterns and resolution strategies
    - Capability gaps and improvement opportunities
    - Success patterns that can be replicated
    - Metadata for prioritizing improvements

    The summary is structured to enable an AI agent to identify patterns across conversations
    and propose data-driven improvements.

    Args:
        client: instructor-patched client
        messages: List of conversation messages

    Returns:
        ConversationSummary object with analysis-optimized summary
    """

    prompt = """
    Create a hybrid summary that excels at both pattern-based (v2) and content-based (v1) search queries.

    CRITICAL: Balance these two needs:
    1. Pattern queries like "conversations where users ask about X" (v2 style)
    2. Content queries like "quantum physics explanation" (v1 style)
    
    Write ONE paragraph (600 chars max) with this structure:
    
    FIRST SENTENCE - Pattern Hook (for v2):
    "Conversation where user [action: asks about/requests/seeks help with/discusses/explores] [specific topic/goal]"
    
    SECOND SENTENCE - Type & Domain (for both):
    "This [type] conversation covers [key technical terms, concepts, tools, or domains]"
    Types: educational Q&A, technical troubleshooting, creative collaboration, task assistance, problem solving
    
    THIRD SENTENCE - Interaction & Content (balanced):
    "The [interaction pattern] includes [specific solutions/information/techniques discussed]"
    Patterns: step-by-step guidance, iterative problem-solving, exploratory discussion, direct Q&A
    
    FOURTH SENTENCE - Searchable Details (for v1):
    Pack in specific technical terms, tools, concepts, and solutions that users might search for.
    
    Balance pattern description with content keywords. Be specific and searchable.
    
    <conversation>
    {% for message in messages %}
        <message role="{{ message.role }}">
            {{ message.content }}
        </message>
    {% endfor %}
    </conversation>
    
    Generate a balanced summary optimized for both pattern and content queries.
    """

    response = await client.chat.completions.create(
        response_model=ConversationSummary,
        messages=[
            {
                "role": "system",
                "content": "You are an expert at analyzing AI conversations for system improvement. Focus on identifying failure patterns, root causes, and specific opportunities for enhancement. Your summaries should enable data-driven decisions about where to invest development effort.",
            },
            {"role": "user", "content": prompt},
        ],
        context={"messages": messages},
        max_retries=3,
    )

    return response
