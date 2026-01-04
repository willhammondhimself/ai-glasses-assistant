# WHAM Meeting Mode

Real-time meeting assistant with tactical suggestions for Halo Frame AR glasses.

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Halo Glasses      │────▶│    Backend API      │
│   (MeetingMode)     │◀────│    (FastAPI)        │
└─────────────────────┘     └─────────────────────┘
         │                          │
         │ WebSocket                │
         │                          ▼
         │                  ┌───────────────────┐
         │                  │  MeetingHandler   │
         │                  │  (router.py)      │
         │                  └───────────────────┘
         │                          │
         │         ┌────────────────┼────────────────┐
         │         ▼                ▼                ▼
         │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
         │   │ Transcription│ │  Context     │ │  Suggestion  │
         │   │ Service      │ │  Manager     │ │  Engine      │
         │   │ (Whisper)    │ │              │ │ (Gemini 2.5) │
         │   └──────────────┘ └──────────────┘ └──────────────┘
         │                          │
         └──────────────────────────┘
```

## Components

| Component | File | Purpose |
|-----------|------|---------|
| **MeetingHandler** | `router.py` | WebSocket message routing and session management |
| **TranscriptionService** | `transcription.py` | Real-time audio → text via Whisper API |
| **ContextManager** | `context.py` | Transcript buffer and context window management |
| **SuggestionEngine** | `suggestions.py` | Gemini 2.5 Pro tactical suggestions |
| **Models** | `models.py` | Data classes for all meeting entities |

## WebSocket Protocol

### Endpoint

```
ws://localhost:8000/ws/meeting/{session_id}
```

### Client → Server Messages

| Type | Payload | Description |
|------|---------|-------------|
| `meeting_start` | `{config: {...}}` | Start a new meeting session |
| `meeting_end` | `{}` | End the meeting session |
| `audio_chunk` | `{audio, duration_ms, sample_rate}` | Audio for transcription |
| `double_tap` | `{}` | Quick suggestion trigger (2-3s) |
| `voice_command` | `{query, audio?}` | Detailed help trigger (3-4s) |
| `ping` | `{}` | Keepalive ping |

### Server → Client Messages

| Type | Payload | Description |
|------|---------|-------------|
| `status` | `{status, message?, session_id?, summary?}` | Status updates |
| `transcript_update` | `{segment: {...}}` | New transcript segment |
| `suggestion` | `{trigger, suggestion, total_latency_ms}` | WHAM's suggestion |
| `error` | `{message}` | Error message |
| `pong` | `{}` | Keepalive response |

### Message Examples

**Start Meeting:**
```json
{
  "type": "meeting_start",
  "config": {
    "meeting_type": "negotiation",
    "participants": ["Will", "Counterpart"],
    "context": "Salary negotiation for senior role",
    "proactive_suggestions": false
  }
}
```

**Audio Chunk:**
```json
{
  "type": "audio_chunk",
  "audio": "UklGRiQAAABXQVZFZm10...",
  "duration_ms": 1000,
  "sample_rate": 16000
}
```

**Double-Tap Response:**
```json
{
  "type": "suggestion",
  "trigger": "double_tap",
  "suggestion": {
    "suggestion": "Consider proposing a phased approach...",
    "type": "tactical_advice",
    "confidence": 0.88,
    "alternatives": ["Ask for clarification...", "Propose milestone-based..."],
    "tactical_notes": "They seem hesitant - this is leverage."
  },
  "total_latency_ms": 1850
}
```

## Audio Format

| Parameter | Value |
|-----------|-------|
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Bit depth | 16-bit PCM |
| Chunk size | 1000ms |
| Encoding | Base64-encoded WAV |

## Latency Benchmarks

| Trigger | Context Size | Target | Model |
|---------|-------------|--------|-------|
| Double-tap | 5 segments (~30s) | 2-3s | Gemini Flash |
| Voice command | Full transcript | 3-4s | Gemini 2.5 Pro |

## Meeting Types

| Type | Focus | Suggestion Style |
|------|-------|-----------------|
| `general` | Balanced | General meeting advice |
| `negotiation` | Leverage & positioning | Tactical/strategic |
| `interview` | Candidate assessment | Question prompts |
| `sales` | Objection handling | Closing techniques |

## Suggestion Types

| Type | Description | Trigger |
|------|-------------|---------|
| `quick_response` | What to say right now | Both |
| `tactical_advice` | Strategic meeting advice | Voice |
| `fact_check` | Verify a claim | Voice |
| `context_fill` | Background on a topic | Voice |
| `negotiation` | Leverage/positioning advice | Both |
| `clarification` | Suggest asking for clarity | Both |

## Running Tests

```bash
# Run all meeting mode tests
pytest backend/tests/test_meeting_mode.py -v

# Run with coverage
pytest backend/tests/test_meeting_mode.py -v --cov=backend/meeting
```

## CLI Test Client

Test meeting mode without Halo glasses hardware:

```bash
# Start the server
uvicorn backend.server:app --reload

# In another terminal, run the test client
python -m backend.meeting.test_client

# Interactive session example:
wham> start negotiation
[STATUS] Meeting started | Session: abc12345

wham> tap
[PROCESSING] Thinking...
[SUGGESTION] (double_tap, 1850ms)
  Consider proposing a phased approach...

wham> ask What leverage do I have here?
[PROCESSING] Processing: What leverage do I have here?
[SUGGESTION] (voice_command, 2340ms)
  Based on the conversation, you have leverage in...

wham> end
[STATUS] Meeting ended
         Duration: 120s | Suggestions: 2
```

## Environment Variables

Required in `.env`:

```bash
OPENAI_API_KEY=sk-...     # For Whisper transcription
GEMINI_API_KEY=AIza...    # For Gemini 2.5 Pro suggestions
```

## Extending Suggestion Types

### 1. Add new type in `models.py`:

```python
class SuggestionType(Enum):
    QUICK_RESPONSE = "quick_response"
    # ... existing types ...
    MY_NEW_TYPE = "my_new_type"  # Add here
```

### 2. Update classification in `suggestions.py`:

```python
def _classify_suggestion(self, suggestion_text: str, context: dict) -> SuggestionType:
    # Add detection logic for the new type
    if "specific_indicator" in suggestion_text.lower():
        return SuggestionType.MY_NEW_TYPE
    # ...
```

### 3. Update prompt templates in `suggestions.py`:

```python
# Add type-specific instructions to the Gemini prompt
def _build_prompt(self, context: dict, trigger: TriggerType, query: str = None):
    # Include guidance for generating MY_NEW_TYPE suggestions
    pass
```

## Display Integration

Meeting mode uses these display colors (from `glasses_client/core/display.py`):

| Color | RGB | Usage |
|-------|-----|-------|
| `meeting` | `(78, 205, 196)` | Teal - general status |
| `suggestion` | `(149, 165, 166)` | Silver - suggestions |
| `recording` | `(255, 107, 107)` | Coral - recording indicator |
| `processing` | `(255, 230, 109)` | Yellow - thinking |
| `negotiation` | `(255, 177, 66)` | Orange - negotiation alerts |
| `alert` | `(255, 107, 129)` | Pink-red - important alerts |

## Halo SDK Integration

See TODO comments in `glasses_client/modes/meeting.py` for integration points:

- `frame.microphone.record()` - Audio streaming
- `frame.imu.wait_for_tap()` - Tap detection
- `frame.voice.check_wake_word()` - Wake word detection
- `frame.display.show_text()` - OLED display
- `frame.audio.speak()` - Bone conduction feedback
