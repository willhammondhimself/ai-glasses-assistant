# AI Glasses Assistant - WHAM

My personal AR glasses project for the Brilliant Labs Halo.

Think Iron Man's JARVIS meets Tony Stark's workshop - but personalized for me.

## What This Does

- **Mental Math Mode**

- **Poker Mode**

- **Code Debug**

- **WHAM Vision**

- **Morning Briefing**

- **Quick Capture**

- **Focus Mode**

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

Of course, this is a personal project, so it's not poerfect software. In my testing, mental math works great, poker analysis works if you have an API key, web dashboard is fully functional without glasses. WHAM Vision scanning is operational with Gemini 2.0 Flash for screen detection.

Still TODO: Halo hardware integration (mine are still shipping), and it's hardcoded to call everyone "sir" and assume you're me.

## Tech Stack

FastAPI backend with WebSockets, Claude for poker/code analysis, Gemini for vision, MicroPython on the glasses. Dashboard is vanilla JS because I didn't want to deal with React for something this small.

## License

It's a personal project, do whatever you want with it. If you're also prepping for quant interviews, try it out!
