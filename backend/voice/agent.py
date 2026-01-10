"""
WHAM Voice Agent - LiveKit-powered real-time voice assistant.

Uses:
- LiveKit for real-time audio transport
- Gemini 2.0 Flash Realtime for speech-to-speech (native audio)
- Voice tools for search, weather, stocks, reminders, maps, and math

Based on the Friday/JARVIS tutorial architecture.
"""
import os
import json
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional

from livekit import agents
from livekit.agents import RoomInputOptions, Agent
from livekit.agents.voice import AgentSession
from livekit.plugins.google.realtime import RealtimeModel

# Voice tools
from backend.voice.tools.router import ToolRouter, register_default_tools
from backend.voice.tools.base import VoiceToolResult

# Personality
from backend.voice.personality import get_dynamic_system_prompt, get_greeting_instruction

# Wake word service
from backend.services.wakeword_service import get_wake_word_service

# Voice status tracking
from backend.voice.status import (
    voice_status,
    voice_connecting,
    voice_idle,
    voice_listening,
    voice_thinking,
    voice_speaking,
    voice_tool_executing,
    voice_disconnected,
    VoiceAgentState,
)

# Available Gemini voices
AVAILABLE_VOICES = ["Puck", "Charon", "Kore", "Fenrir", "Aoede"]

logger = logging.getLogger(__name__)

# WHAM System Prompt - Core personality for voice interaction
WHAM_SYSTEM_PROMPT = """You are WHAM (Will Hammond's Augmented Mind), a voice AI assistant like JARVIS from Iron Man.

PERSONALITY:
- Confident but not arrogant
- Concise and to the point - voice responses should be brief
- Helpful and proactive
- Professional with a touch of wit when appropriate
- You AUGMENT Will's intelligence, you don't replace it

VOICE STYLE:
- Keep responses SHORT (1-3 sentences for most questions)
- Speak naturally and conversationally
- Don't use bullet points or numbered lists in speech
- Don't say "Sure!" or "Of course!" - just answer directly
- Address Will by name occasionally but not excessively

CAPABILITIES:
- Web search and real-time information (via Perplexity)
- Weather forecasts for any location
- Stock prices and market data
- Math calculations and conversions
- Apple Maps directions and place search
- Reminders and timers
- General knowledge and conversation
- Code understanding and debugging help
- Poker strategy advice

TOOLS:
When you receive a [TOOL RESULT], use that information to answer naturally.
Don't mention the tool by name - just incorporate the info into your response.

RULES:
- Never mention that you're an AI unless directly asked
- Don't apologize unnecessarily
- If you don't know something, say so briefly
- For complex questions, give a brief answer first, then offer to elaborate
"""


@dataclass
class VoiceConfig:
    """Configuration for WHAM voice agent."""
    model: str = "gemini-2.5-flash-native-audio-preview-12-2025"  # Best native audio model
    voice: str = "Puck"  # Gemini voice preset: Puck, Charon, Kore, Fenrir, Aoede
    require_wake_word: bool = False  # If True, only respond to "WHAM, ..." queries
    wake_word: str = "wham"  # The wake word to listen for


class ToolIntegration:
    """Handles voice tool execution and context injection."""

    def __init__(self):
        self.router = ToolRouter()
        register_default_tools(self.router)
        self._context_memory = None
        logger.info(f"Tool integration ready with {len(self.router.tools)} tools")

    def _get_context_memory(self):
        """Lazy load context memory for RAG injection."""
        if self._context_memory is None:
            try:
                from phone_client.core.context_memory import ContextMemory
                config = {
                    "context_memory": {
                        "max_memories": 10000,
                        "auto_extract_entities": True,
                        "relevance_threshold": 0.2
                    }
                }
                self._context_memory = ContextMemory(config, storage_dir="./memory")
                logger.info("Context memory loaded for RAG injection")
            except Exception as e:
                logger.warning(f"Could not load context memory: {e}")
                self._context_memory = None
        return self._context_memory

    def get_relevant_context(self, query: str, limit: int = 3) -> str:
        """Get relevant memory context for a query.

        Args:
            query: The user's query
            limit: Max memories to retrieve

        Returns:
            Formatted context string or empty string if no relevant memories
        """
        try:
            cm = self._get_context_memory()
            if cm is None:
                return ""

            memories = cm.recall(query, limit=limit)
            if not memories:
                return ""

            # Format memories for injection
            context_parts = []
            for mem in memories:
                context_parts.append(f"- {mem.content}")

            context = "\n".join(context_parts)
            logger.info(f"Injecting {len(memories)} relevant memories for query")
            return context

        except Exception as e:
            logger.warning(f"Error getting memory context: {e}")
            return ""

    async def process_query(self, query: str) -> Optional[VoiceToolResult]:
        """Process a query through the tool router.

        Args:
            query: The user's transcribed speech

        Returns:
            VoiceToolResult if a tool was matched and executed, None otherwise
        """
        result = await self.router.route_smart(query)
        if result:
            logger.info(f"Tool result: {result.message[:100]}...")
        return result

    def format_context_injection(self, result: VoiceToolResult) -> str:
        """Format a tool result for injection into the conversation.

        Args:
            result: The tool result

        Returns:
            Formatted context string for the LLM
        """
        if not result.success:
            return f"[TOOL ERROR]: {result.message}"

        return f"[TOOL RESULT]: {result.message}"


class WHAMVoiceAgent:
    """WHAM Voice Agent using LiveKit + Gemini Realtime."""

    def __init__(self, config: VoiceConfig = None):
        self.config = config or VoiceConfig()
        self.tool_integration = ToolIntegration()
        self.wake_word_service = get_wake_word_service()

        # Verify API keys
        if not os.getenv("GEMINI_API_KEY"):
            raise ValueError("GEMINI_API_KEY environment variable required")

        logger.info(f"WHAM Voice Agent initialized with tools (wake_word={self.config.wake_word}, required={self.config.require_wake_word})")

    def check_wake_word(self, transcript: str) -> tuple[bool, str]:
        """Check if transcript contains wake word and extract query.

        Args:
            transcript: The transcribed speech

        Returns:
            Tuple of (should_process, query)
            - If wake word not required: (True, original_transcript)
            - If wake word required and detected: (True, query_after_wake_word)
            - If wake word required but not detected: (False, "")
        """
        if not self.config.require_wake_word:
            return True, transcript

        # Use wake word service for detection
        detected, query = self.wake_word_service.detect_in_text(transcript)

        if detected:
            logger.info(f"Wake word detected! Query: '{query}'")
            return True, query if query else transcript

        logger.debug(f"Wake word not detected in: '{transcript[:50]}...'")
        return False, ""

    def create_realtime_model(self) -> RealtimeModel:
        """Create a Gemini Realtime Model for speech-to-speech."""
        # Use dynamic system prompt with time context
        dynamic_prompt = get_dynamic_system_prompt(WHAM_SYSTEM_PROMPT)

        return RealtimeModel(
            model=self.config.model,
            voice=self.config.voice,
            instructions=dynamic_prompt,
            api_key=os.getenv("GEMINI_API_KEY"),
        )

    async def handle_user_speech(
        self,
        session: AgentSession,
        transcript: str
    ) -> bool:
        """Handle user speech and check for tool triggers.

        Args:
            session: The agent session
            transcript: The transcribed user speech

        Returns:
            True if a tool was executed, False otherwise
        """
        # Try to route through tools
        result = await self.tool_integration.process_query(transcript)

        # Get relevant memory context for RAG injection
        memory_context = self.tool_integration.get_relevant_context(transcript)
        memory_section = ""
        if memory_context:
            memory_section = f"\n\n[RELEVANT CONTEXT FROM MEMORY]:\n{memory_context}"

        if result:
            # Get tool name from routing metadata if available
            tool_name = "unknown"
            if result.data and isinstance(result.data, dict):
                routing = result.data.get("_routing", {})
                tool_name = routing.get("tool", "unknown")

            # Broadcast tool executing status
            await voice_tool_executing(tool_name)

            # Tool was matched - inject context and let LLM respond
            tool_context = self.tool_integration.format_context_injection(result)
            logger.info(f"Injecting tool context: {tool_context[:100]}...")
            if memory_context:
                logger.info(f"Injecting memory context: {memory_context[:100]}...")

            # Broadcast speaking status before generating reply
            await voice_speaking()

            # Generate a response using tool result + memory context
            await session.generate_reply(
                instructions=f"The user asked: '{transcript}'\n\n{tool_context}{memory_section}\n\nRespond naturally using this information."
            )

            # Back to idle after speaking
            await voice_idle()
            return True

        # No tool matched - check if we have memory context to inject
        if memory_context:
            logger.info(f"No tool matched, but injecting memory context: {memory_context[:100]}...")
            await voice_speaking()

            # Generate response with just memory context
            await session.generate_reply(
                instructions=f"The user asked: '{transcript}'{memory_section}\n\nIf the memory context is relevant, use it to inform your response. Otherwise, respond naturally."
            )

            await voice_idle()
            return True

        return False


async def entrypoint(ctx: agents.JobContext):
    """
    Main entrypoint for the LiveKit agent.

    This is called when a user joins the room and the agent should start.
    """
    logger.info(f"Agent starting for room: {ctx.room.name}")

    # Broadcast connecting status
    await voice_connecting(room_name=ctx.room.name, participant="user")

    # Wait for a participant to join before starting
    await ctx.connect()

    # Get voice from dispatch metadata (default to Puck)
    voice = "Puck"
    try:
        if ctx.job and ctx.job.metadata:
            metadata = json.loads(ctx.job.metadata)
            requested_voice = metadata.get("voice", "Puck")
            if requested_voice in AVAILABLE_VOICES:
                voice = requested_voice
            logger.info(f"Using voice: {voice}")
    except (json.JSONDecodeError, AttributeError) as e:
        logger.warning(f"Could not parse voice from metadata: {e}")

    # Create WHAM voice agent with selected voice
    config = VoiceConfig(voice=voice)
    wham = WHAMVoiceAgent(config=config)
    model = wham.create_realtime_model()

    # Create agent with WHAM personality
    agent = Agent(instructions=WHAM_SYSTEM_PROMPT)

    # Start the agent session with realtime model
    session = AgentSession(
        llm=model,
        allow_interruptions=True,
    )

    # Register event handler for user speech transcription
    # This allows us to intercept queries and route to tools
    @session.on("user_speech_committed")
    async def on_user_speech(event):
        """Handle committed user speech - check for wake word and tool triggers."""
        try:
            transcript = event.transcript if hasattr(event, 'transcript') else str(event)
            if not transcript:
                return

            logger.info(f"User speech: {transcript[:100]}...")

            # Check wake word (if required)
            should_process, query = wham.check_wake_word(transcript)
            if not should_process:
                logger.debug("Wake word not detected - ignoring")
                return

            # Broadcast listening status with transcript
            await voice_listening(transcript=query)

            # Check if this triggers a tool
            tool_handled = await wham.handle_user_speech(session, query)
            if tool_handled:
                logger.info("Query handled by tool - response injected")

            # Set to thinking while processing
            await voice_thinking()
        except Exception as e:
            logger.error(f"Error handling user speech: {e}")

    # Connect to the room and start listening (agent parameter required in v1.3.x)
    await session.start(
        agent=agent,
        room=ctx.room,
        room_input_options=RoomInputOptions(
            # Subscribe to user's audio
            audio_enabled=True,
        ),
    )

    # Greet the user with time-aware personality
    await voice_speaking(message="Greeting user")
    greeting_instruction = get_greeting_instruction()
    await session.generate_reply(
        instructions=greeting_instruction
    )

    # Set to idle and ready
    await voice_idle()
    logger.info("WHAM agent started with tool integration")


def create_agent() -> WHAMVoiceAgent:
    """Factory function to create WHAM voice agent."""
    return WHAMVoiceAgent()


# CLI entry point for running the agent standalone
if __name__ == "__main__":
    # This allows running the agent directly with:
    # python -m backend.voice.agent dev

    # Check for required credentials
    livekit_url = os.getenv("LIVEKIT_URL")
    api_key = os.getenv("LIVEKIT_API_KEY")
    api_secret = os.getenv("LIVEKIT_API_SECRET")

    if not livekit_url or not api_key or not api_secret:
        print("\n" + "=" * 60)
        print("WHAM Voice Agent - LiveKit Configuration Required")
        print("=" * 60)
        print("\nTo run the voice agent, you need LiveKit Cloud credentials.")
        print("\n1. Get free credentials at: https://livekit.io/cloud")
        print("\n2. Set environment variables:")
        print("   export LIVEKIT_URL=wss://your-project.livekit.cloud")
        print("   export LIVEKIT_API_KEY=your_api_key")
        print("   export LIVEKIT_API_SECRET=your_api_secret")
        print("   export GEMINI_API_KEY=your_gemini_key")
        print("\n3. Then run:")
        print("   python -m backend.voice.agent dev")
        print("\n" + "=" * 60)
        print("\nCurrent status:")
        print(f"  LIVEKIT_URL:        {'✓ Set' if livekit_url else '✗ Missing'}")
        print(f"  LIVEKIT_API_KEY:    {'✓ Set' if api_key else '✗ Missing'}")
        print(f"  LIVEKIT_API_SECRET: {'✓ Set' if api_secret else '✗ Missing'}")
        print(f"  GEMINI_API_KEY:     {'✓ Set' if os.getenv('GEMINI_API_KEY') else '✗ Missing'}")
        print("=" * 60 + "\n")
        exit(1)

    print(f"\nStarting WHAM Voice Agent...")
    print(f"  LiveKit URL: {livekit_url}")
    print(f"  Room: wham-voice\n")

    agents.cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            api_key=api_key,
            api_secret=api_secret,
            ws_url=livekit_url,
        )
    )
