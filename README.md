# AI Glasses Assistant - WHAM (Will's Helpful Assistant Module)

My personal AR glasses project for the Brilliant Labs Halo. Built this to prep for quant interviews at Jane Street/Cit Sec/HRT because staring at flashcards got old fast.

Think Iron Man's JARVIS meets Tony Stark's workshop - but personalized for my workflow.

## What This Actually Does

**Mental Math Mode** - Throws arithmetic at you with quant-trader mental-math interview time pressure. D1 problems give you 2 seconds, D5 gives you 20. Miss the window and WHAM will let you know about it (politely, but firmly).

**Poker Mode** - Point at cards, get GTO analysis. Uses Claude to break down preflop ranges, pot odds, bet sizing. Useful for home games when you're trying to figure out if that river shove is a bluff.

**Code Debug** - Still WIP but the idea is to look at code on a screen and have WHAM explain what's broken. Currently does basic static analysis (catches bare excepts, eval() usage, the usual).

**WHAM Vision** - Background vision that notices when you're staring at an equation or code snippet for 1.5+ seconds, then offers to help. Tries not to be annoying about it.

**Morning Briefing** - "Good morning sir, it's 47 degrees and you have Stochastic Calc at 10." That kind of thing.

**Quick Capture** - Say something, it remembers it. Tags and categorizes automatically so I don't have to.

**Focus Mode** - Pomodoro but WHAM tracks it and roasts you if you keep getting distracted.

## The WHAM Personality

I got tired of robotic AI responses, so everything goes through a personality layer that makes it sound like you have a personal AI assistant in your ear:

- "2.3s - that's Jane Street caliber, sir."
- "The correct answer was 847. Onward."
- "Twenty consecutive. Shall I notify the trading firms?"

It tracks your session, knows when you're warming up vs in the zone vs struggling, and adjusts accordingly. The streak system has tiers (bronze through legendary) because gamification works on me.

## Project Structure

```
backend/
    wham/            # Personality layer - templates, context tracking, performance analysis
    dashboard/       # Web dashboard API
    hud/             # Display components - colors, progress bars, layouts
    vision/          # Background scanner - detection, power management
    websocket/       # Real-time handlers for each mode
    poker/           # GTO analysis engine
    quant/           # Mental math problem generation
    server.py        # FastAPI entry point

phone_client/        # Mobile/local client with offline capabilities
    wham/            # WHAM personality implementation
    modes/           # Different operational modes
    integrations/    # External service connectors (Calendar, Weather, Notion)
    halo/            # Halo glasses SDK integration
    hud/             # HUD rendering

desktop_capture/     # Desktop quick capture app with system tray

frontend/            # Web dashboard (works without glasses)
```

## Running It

```bash
pip install -r requirements.txt
python -m uvicorn backend.server:app --reload

# .env needs these
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...  # for WHAM Vision

# Dashboard at http://localhost:8000/dashboard.html
```

The glasses connect over WebSocket. Protocol is pretty simple:

```json
// Start mental math
{"action": "start", "difficulty": 2}

// Submit answer
{"action": "answer", "answer": "847"}

// Get WHAM feedback
{"type": "result", "correct": true, "time_ms": 2340, "wham": {"feedback": "Well executed, sir. 2.34s."}}
```

## Hardware

Built for the [Brilliant Labs Halo](https://brilliant.xyz/products/halo) - 640x400 OLED, 20 degree FOV, full RGB. The HUD code generates render data that maps to their display format.

Should work with any AR glasses that can run Python and connect to a backend, but the display layouts are sized for the Halo specifically.

**No glasses?** Use the web dashboard at `/dashboard.html` to access all WHAM features.

## Why

I'm a physics/math/CS student prepping for quant trading interviews. The mental math rounds at these places are brutal- you get like 8 minutes to answer 80 arithmetic questions. Drilling on a laptop got boring. Drilling while walking around campus with glasses is slightly less boring.

The poker thing started because I play home games and wanted to actually understand ranges instead of just vibing. WHAM Vision was just me wanting to feel like Tony Stark with an AI assistant honestly.

## Status

Personal project, not production software. Mental math works great, poker analysis works if you have the API key, web dashboard is fully functional without glasses. WHAM Vision scanning is operational with Gemini 2.0 Flash for screen detection.

Still TODO: better Halo hardware integration, and it's hardcoded to call everyone "sir" and assume you're me.

## Tech Stack

FastAPI backend with WebSockets, Claude for poker/code analysis, Gemini for vision, MicroPython on the glasses. Dashboard is vanilla JS because I didn't want to deal with React for something this small.

## License

It's a personal project, do whatever you want with it. If you're also prepping for quant interviews, try it out!
