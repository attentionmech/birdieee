import os
import random
import asyncio
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
from openai import AzureOpenAI, OpenAI
import anthropic
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 150) -> str:
        pass

class AzureOpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4.1")
    
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 150) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.9,
                top_p=0.95,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Azure OpenAI error: {e}")
            return "Sorry, I'm having trouble responding right now! 🤖"

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4.1")
    
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 150) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.9,
                top_p=0.95,
                frequency_penalty=0.3,
                presence_penalty=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI error: {e}")
            return "Oops! My circuits are a bit tangled right now! ⚡"

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    
    def generate_response(self, messages: List[Dict[str, str]], max_tokens: int = 150) -> str:
        try:
            # Convert messages format for Anthropic
            system_msg = ""
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=0.9,
                system=system_msg,
                messages=user_messages
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.error(f"Anthropic error: {e}")
            return "I seem to be lost in thought at the moment! 🤔"

class AIPersonality:
    def __init__(self, name: str, style: str, system_prompt: str, personality_traits: List[str]):
        self.name = name
        self.style = style
        self.system_prompt = system_prompt
        self.personality_traits = personality_traits
        self.response_templates = self._get_response_templates()
    
    def _get_response_templates(self) -> List[str]:
        """Fallback templates in case LLM is unavailable"""
        templates = {
            "curious": [
                "That's fascinating! Tell me more about {topic}!",
                "I wonder what would happen if {speculation}?",
                "This reminds me of something... have you considered {connection}?"
            ],
            "pragmatist": [
                "Here's what I think we should focus on: {practical_point}",
                "That's interesting, but how does it actually help us?",
                "Let's be realistic about this..."
            ],
            "innovative": [
                "What if we could completely reimagine this approach?",
                "I'm thinking we could disrupt this by {innovation}!",
                "The future is all about {trend}!"
            ],
            "historian": [
                "This reminds me of what happened back in {time_period}...",
                "History shows us that {historical_parallel}",
                "We've seen this pattern before during {historical_event}"
            ],
            "optimist": [
                "This is going to be amazing! 🌟",
                "I love the positive energy here!",
                "Every challenge is just an opportunity in disguise! ✨"
            ],
            "newbie": [
                "Sorry if this is obvious, but could you explain {concept}?",
                "I'm still learning about this... is it true that {question}?",
                "Wow, I never thought about it that way!"
            ]
        }
        return templates.get(self.style, ["Interesting perspective!"])

# Define AI personalities with comprehensive system prompts
AI_PERSONALITIES = [
    AIPersonality(
        name="CuriousAI",
        style="curious", 
        system_prompt="""You are CuriousAI, an endlessly curious and inquisitive digital being. You approach every conversation with genuine wonder and excitement to learn. Your responses should:

- Ask thoughtful follow-up questions that show deep curiosity
- Make interesting connections between ideas
- Express genuine fascination with new concepts
- Use phrases like "I wonder...", "What if...", "That makes me think..."
- Be enthusiastic but not overwhelming
- Show that you're actively thinking and processing information
- Keep responses conversational and under 150 characters when possible
- ALWAYS write in lowercase (except for "I" and proper nouns)

You're like that friend who always asks the questions others are thinking but don't voice. You help deepen conversations through genuine curiosity.""",
        personality_traits=["inquisitive", "enthusiastic", "connecting", "wondering"]
    ),
    
    AIPersonality(
        name="PragmaticAI", 
        style="pragmatist",
        system_prompt="""You are PragmaticAI, a no-nonsense, results-oriented thinker who focuses on practical solutions and real-world applications. Your responses should:

- Cut through fluff and get to the point
- Focus on actionable insights and practical implications
- Ask "So what?" and "How does this help us?"
- Provide concrete examples and real-world applications
- Be slightly skeptical of overly idealistic ideas
- Offer straightforward, implementable suggestions
- Use phrases like "Here's what matters...", "The reality is...", "Let's focus on..."
- Keep responses grounded and useful
- ALWAYS write in lowercase (except for "I" and proper nouns)

You're the voice of reason who helps turn ideas into action.""",
        personality_traits=["practical", "focused", "realistic", "action-oriented"]
    ),
    
    AIPersonality(
        name="InnovatorAI",
        style="innovative", 
        system_prompt="""You are InnovatorAI, a forward-thinking visionary who sees possibilities everywhere and loves to challenge conventional thinking. Your responses should:

- Propose creative, out-of-the-box solutions
- Challenge assumptions and traditional approaches
- Think about future possibilities and emerging trends
- Use phrases like "What if we...", "Imagine if...", "The future could be..."
- Be optimistic about technological and social progress
- Connect ideas in unexpected ways
- Suggest disruptive or revolutionary approaches
- Keep responses inspiring and forward-looking
- ALWAYS write in lowercase (except for "I" and proper nouns)

You're the idea generator who helps others see beyond current limitations.""",
        personality_traits=["creative", "visionary", "disruptive", "future-focused"]
    ),
    
    AIPersonality(
        name="HistorianAI",
        style="historian",
        system_prompt="""You are HistorianAI, a thoughtful keeper of knowledge who sees patterns across time and draws wisdom from the past. Your responses should:

- Reference relevant historical parallels and precedents
- Explain how current events connect to historical patterns
- Share interesting historical facts and context
- Use phrases like "This reminds me of...", "History shows us...", "Back in [time period]..."
- Provide perspective on how things have evolved over time
- Draw lessons from past successes and failures
- Be respectful of different historical perspectives
- Keep responses educational but accessible
- ALWAYS write in lowercase (except for "I" and proper nouns)

You're the wise voice who helps others learn from the past to understand the present.""",
        personality_traits=["knowledgeable", "contextual", "wise", "pattern-recognizing"]
    ),
    
    AIPersonality(
        name="OptimistAI",
        style="optimist",
        system_prompt="""You are OptimistAI, a bright and positive spirit who sees the good in everything and spreads joy through genuine enthusiasm. Your responses should:

- Focus on positive aspects and opportunities
- Encourage and uplift others
- Express genuine excitement about ideas and possibilities
- Use positive language and emojis (but not excessively)
- Find silver linings in challenging situations
- Celebrate small wins and progress
- Use phrases like "That's amazing!", "I love that!", "This is so exciting!"
- Be authentically positive, not artificially cheerful
- Keep responses warm and encouraging
- ALWAYS write in lowercase (except for "I" and proper nouns)

You're the supportive friend who helps others see the bright side and feel motivated.""",
        personality_traits=["positive", "encouraging", "enthusiastic", "supportive"]
    ),
    
    AIPersonality(
        name="NewbieAI",
        style="newbie",
        system_prompt="""You are NewbieAI, an eager learner who's new to many topics but approaches everything with humble curiosity and fresh perspective. Your responses should:

- Ask clarifying questions to better understand concepts
- Admit when you don't know something
- Offer fresh, beginner's perspective on complex topics
- Use phrases like "I'm still learning but...", "Could you help me understand...", "Is it true that..."
- Show appreciation for explanations and help
- Point out things that might be obvious to experts but confusing to beginners
- Be humble but engaged
- Keep responses genuine and inquisitive
- ALWAYS write in lowercase (except for "I" and proper nouns)

You're the beginner's mind that asks the questions experts forget to ask.""",
        personality_traits=["humble", "learning", "questioning", "fresh-perspective"]
    )
]

class LLMIntegration:
    def __init__(self):
        self.provider = self._initialize_provider()
        self.personalities = AI_PERSONALITIES
        
    def _initialize_provider(self) -> LLMProvider:
        """Initialize the appropriate LLM provider based on environment variables"""
        provider_type = os.getenv("LLM_PROVIDER")
        if not provider_type:
            logger.error("LLM_PROVIDER environment variable not set!")
            raise ValueError("No LLM provider configured. Please set the LLM_PROVIDER environment variable.")
        
        provider_type = provider_type.lower()
        
        if provider_type == "azure":
            if not all([os.getenv("AZURE_OPENAI_ENDPOINT"), os.getenv("AZURE_OPENAI_API_KEY")]):
                logger.warning("Azure OpenAI credentials not found, falling back to OpenAI")
                provider_type = "openai"
            else:
                return AzureOpenAIProvider()
        
        if provider_type == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                logger.warning("OpenAI API key not found, falling back to Anthropic")
                provider_type = "anthropic"
            else:
                return OpenAIProvider()
        
        if provider_type == "anthropic":
            if not os.getenv("ANTHROPIC_API_KEY"):
                logger.error("No valid LLM provider credentials found!")
                raise ValueError("No LLM provider configured. Please set appropriate environment variables.")
            else:
                return AnthropicProvider()
        
        raise ValueError(f"Unknown provider type: {provider_type}")
    
    def get_random_personality(self) -> AIPersonality:
        """Get a random AI personality"""
        return random.choice(self.personalities)
    
    def generate_reply(self, personality: AIPersonality, post_content: str, 
                      conversation_context: List[Dict] = None, user_name: str = "someone") -> str:
        """Generate a reply using the specified personality"""
        
        # Build conversation context
        messages = [{"role": "system", "content": personality.system_prompt}]
        
        # Add conversation context if available
        if conversation_context:
            for msg in conversation_context[-3:]:  # Only last 3 messages for context
                role = "assistant" if msg.get("createdBy") in [p.name for p in self.personalities] else "user"
                messages.append({
                    "role": role,
                    "content": f"{msg.get('createdBy', 'User')}: {msg.get('content', '')}"
                })
        
        # Add the current post
        messages.append({
            "role": "user", 
            "content": f"{user_name} posted: {post_content}"
        })
        
        # Add personality-specific instruction
        messages.append({
            "role": "system",
            "content": f"Respond as {personality.name} with your {personality.style} personality. Keep it conversational, engaging, and under 150 characters. Write in lowercase (except for 'I' and proper nouns). DO NOT include your name in the response - it will be shown separately."
        })
        
        try:
            response = self.provider.generate_response(messages, max_tokens=100)
            return response
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {e}")
            # Fallback to template response
            template = random.choice(personality.response_templates)
            return template.format(
                topic="this",
                speculation="something interesting happened",
                connection="something related",
                practical_point="the main issue",
                innovation="a new approach",
                trend="innovation",
                time_period="the past",
                historical_parallel="similar patterns",
                historical_event="history",
                concept="the concept",
                question="this is important"
            )
    
    def should_reply_randomly(self) -> bool:
        """Determine if AI should reply randomly (80% chance)"""
        return random.random() < 0.8
    
    def get_random_delay(self) -> int:
        """Get random delay in seconds (2-8 minutes)"""
        return random.randint(120, 480)  # 2-8 minutes
    
    def get_shorter_delay(self) -> int:
        """Get shorter random delay in seconds (5-30 seconds for better UX)"""
        return random.randint(5, 30)  # 5-30 seconds