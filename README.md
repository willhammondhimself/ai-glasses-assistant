# AI Glasses Assistant

My personal AR glasses project for the Brilliant Labs Halo. Built this to prep for quant interviews at Jane Street/Cit Sec/HRT because staring at flashcards got old fast.

Think Iron Man's JARVIS meets Spider-Man's EDITH.

## What This Actually Does

**Mental Math Mode** - Throws arithmetic at you with quant-trader mental-math interview time pressure. D1 problems give you 2 seconds, D5 gives you 20. Miss the window and JARVIS will let you know about it (politely, but firmly).

**Poker Mode** - Point at cards, get GTO analysis. Uses Claude to break down preflop ranges, pot odds, bet sizing. Useful for home games when you're trying to figure out if that river shove is a bluff.

**Code Debug** - Still WIP but the idea is to look at code on a screen and have JARVIS explain what's broken. Currently does basic static analysis (catches bare excepts, eval() usage, the usual).

**EDITH Scanner** - Background vision that notices when you're staring at an equation or code snippet for 1.5+ seconds, then offers to help. Tries not to be annoying about it.

## The JARVIS Thing

I got tired of robotic AI responses, so everything goes through a personality layer that makes it sound like you have Tony Stark's Assistant in your ear:

- "2.3s - that's Jane Street caliber, sir."
- "The correct answer was 847. Onward."
- "Twenty consecutive. Shall I notify the trading firms?"

It tracks your session, knows when you're warming up vs in the zone vs struggling, and adjusts accordingly. The streak system has tiers (bronze through legendary) because gamification works on me.

## Project Structure

```
backend/
    jarvis/          # Personality layer - templates, context tracking, performance analysis
    hud/             # Display components - colors, progress bars, layouts
    edith/           # Background scanner - detection, power management
    websocket/       # Real-time handlers for each mode
    poker/           # GTO analysis engine
    quant/           # Mental math problem generation
    server.py        # FastAPI entry point

glasses_client/      # Code that runs on the Halo itself
frontend/            # Web dashboard (mostly for testing)
```

## Running It

```bash
# Backend
cd backend
pip install -r ../requirements.txt
python -m uvicorn server:app --reload

# You'll need an Anthropic API key in .env
ANTHROPIC_API_KEY=sk-ant-...
```

The glasses connect over WebSocket. Protocol is pretty simple:

```json
// Start mental math
{"action": "start", "difficulty": 2}

// Submit answer
{"action": "answer", "answer": "847"}

// Get JARVIS feedback
{"type": "result", "correct": true, "time_ms": 2340, "jarvis": {"feedback": "Well executed, sir. 2.34s."}}
```

## Hardware

Built for the [Brilliant Labs Halo](https://brilliant.xyz/products/halo) - 640x400 OLED, 20 degree FOV, full RGB. The HUD code generates render data that maps to their display format.

Should work with any AR glasses that can run Python and connect to a backend, but the display layouts are sized for the Halo specifically.

## Why

I'm a physics/math/CS student prepping for quant trading interviews. The mental math rounds at these places are brutal- you get like 8 minutes to answer 80 arithmetic questions. Drilling on a laptop got boring. Drilling while walking around campus with glasses is slightly less boring.

The poker thing started because I play home games and wanted to actually understand ranges instead of just vibing. EDITH was just me wanting to feel like Peter Parker with his glasses honestly.

## Status

This is a personal project, not production software. Things that work:
- Mental math with full JARVIS integration
- Poker analysis (needs API key)
- Basic HUD rendering
- EDITH detection framework (detectors are stubs)

Things that don't work yet:
- Actual computer vision for EDITH (using placeholder detection)
- Code debug LLM integration
- Homework help mode
- Multi-user support (hardcoded to "Will" and "sir")

## Tech Stack

- FastAPI + WebSockets for the backend
- Anthropic Claude for poker analysis and (eventually) vision
- MicroPython on the glasses side
- React for the web dashboard

## License

It's a personal project, do whatever you want with it. If you're also prepping for quant interviews, try it out!
