"""
WHAM Desktop Quick Capture - System Tray Application.
Global hotkey for quick capture without opening browser.

Features:
- System tray icon with live status
- Global hotkey (Ctrl+Shift+W / Cmd+Shift+W)
- Quick capture with auto-save drafts
- Voice recording with waveform visualization
- Morning briefing notifications
- Focus mode integration
- Recent captures menu
"""
from __future__ import annotations

import json
import struct
import sys
import threading
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add phone_client to path
PHONE_CLIENT_PATH = Path(__file__).parent.parent / "phone_client"
sys.path.insert(0, str(PHONE_CLIENT_PATH))

# GUI imports
try:
    import tkinter as tk
    from tkinter import ttk
    HAS_TK = True
except ImportError:
    HAS_TK = False

# System tray
try:
    import pystray
    from PIL import Image, ImageDraw
    HAS_TRAY = True
except ImportError:
    HAS_TRAY = False

# Global hotkeys
try:
    import keyboard
    HAS_KEYBOARD = True
except ImportError:
    HAS_KEYBOARD = False

# Audio recording
try:
    import pyaudio
    import wave
    HAS_AUDIO = True
except ImportError:
    HAS_AUDIO = False

# Whisper for transcription
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

# HTTP requests
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# YAML config
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Draft file location
DRAFTS_FILE = PHONE_CLIENT_PATH / "captures" / "draft.json"


class QuickCaptureApp:
    """
    Desktop Quick Capture Application.

    Features:
    - System tray icon with live status
    - Global hotkey (Ctrl+Shift+W / Cmd+Shift+W)
    - Popup text input for quick capture with auto-save drafts
    - Voice recording with waveform visualization
    - Morning briefing notifications
    - Focus mode from tray menu
    - Recent captures menu
    - Auto-sync to WHAM API
    """

    API_BASE = "http://localhost:8000/dashboard/api"
    HOTKEY = "ctrl+shift+w"  # Platform-specific
    AUDIO_RATE = 16000
    AUDIO_CHUNK = 1024
    AUDIO_FORMAT = pyaudio.paInt16 if HAS_AUDIO else None

    def __init__(self):
        self.running = False
        self.tray_icon: Optional[pystray.Icon] = None
        self.capture_window: Optional[tk.Tk] = None
        self.recording_window: Optional[tk.Tk] = None
        self.whisper_model = None

        # Recording state
        self.is_recording = False
        self.audio_frames = []
        self.audio_stream = None
        self.audio_interface = None
        self.waveform_canvas = None

        # Scheduler state
        self.scheduler_running = False
        self.scheduler_thread = None

        # Morning briefing state
        self.morning_assistant = None
        self.briefing_shown_today = False
        self._last_briefing = None
        self._last_briefing_date = None

        # Focus mode state
        self.focus_mode = None

        # Config cache
        self._config = None

        # Platform detection
        self.is_mac = sys.platform == "darwin"
        if self.is_mac:
            self.HOTKEY = "command+shift+w"

        print("WHAM Quick Capture initializing...")
        self._check_dependencies()

    def _check_dependencies(self):
        """Check and report missing dependencies."""
        missing = []

        if not HAS_TK:
            missing.append("tkinter (built-in)")
        if not HAS_TRAY:
            missing.append("pystray, pillow (pip install pystray pillow)")
        if not HAS_KEYBOARD:
            missing.append("keyboard (pip install keyboard)")
        if not HAS_AUDIO:
            missing.append("pyaudio (pip install pyaudio)")
        if not HAS_WHISPER:
            missing.append("openai-whisper (pip install openai-whisper)")
        if not HAS_REQUESTS:
            missing.append("requests (pip install requests)")
        if not HAS_YAML:
            missing.append("pyyaml (pip install pyyaml)")

        if missing:
            print("\n⚠️  Missing optional dependencies:")
            for dep in missing:
                print(f"   - {dep}")
            print("\nBasic functionality available. Install missing for full features.\n")

    def _load_config(self) -> dict:
        """Load config from phone_client/config.yaml."""
        if self._config is not None:
            return self._config

        config_path = PHONE_CLIENT_PATH / "config.yaml"
        if config_path.exists() and HAS_YAML:
            try:
                with open(config_path) as f:
                    self._config = yaml.safe_load(f)
                    return self._config
            except Exception as e:
                print(f"Config load error: {e}")

        self._config = {}
        return self._config

    # =========================================================================
    # TRAY ICON AND MENU
    # =========================================================================

    def create_tray_icon(self) -> Image.Image:
        """Create the system tray icon."""
        size = 64
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw a stylized "W" for WHAM
        draw.ellipse([4, 4, size-4, size-4], fill=(99, 102, 241))
        draw.text((size//2 - 12, size//2 - 14), "W", fill="white")

        return image

    def _get_status_text(self) -> str:
        """Generate current status string for menu header."""
        parts = []

        if self.is_recording:
            parts.append("Recording...")

        if self.focus_mode:
            session = self.focus_mode.get_current_session()
            if session and session.state.value != "idle":
                remaining = self.focus_mode.get_time_remaining()
                mins, secs = divmod(remaining, 60)
                parts.append(f"Focus: {mins:02d}:{secs:02d}")

        if not parts:
            parts.append("Ready")

        return "WHAM - " + " | ".join(parts)

    def _get_focus_status(self) -> str:
        """Get focus mode status text for submenu."""
        if not self.focus_mode:
            return "Status: Idle"

        session = self.focus_mode.get_current_session()
        if not session or session.state.value == "idle":
            return "Status: Idle"

        remaining = self.focus_mode.get_time_remaining()
        mins, secs = divmod(remaining, 60)
        return f"Focusing: {mins:02d}:{secs:02d}"

    def _focus_active(self) -> bool:
        """Check if focus session is active."""
        if not self.focus_mode:
            return False
        session = self.focus_mode.get_current_session()
        return session is not None and session.state.value != "idle"

    def _create_recent_menu(self):
        """Create submenu with recent captures."""
        try:
            from modes.quick_capture import QuickCapture

            config = self._load_config()
            storage_dir = str(PHONE_CLIENT_PATH / "captures")
            capture = QuickCapture(config, storage_dir=storage_dir)
            recent = capture.get_recent(limit=5)

            if not recent:
                return pystray.Menu(
                    pystray.MenuItem("No recent captures", None, enabled=False)
                )

            items = []
            for cap in recent:
                preview = cap.content[:35] + "..." if len(cap.content) > 35 else cap.content

                def make_handler(c):
                    return lambda icon, item: self._show_capture_detail(c)

                items.append(pystray.MenuItem(
                    f"{cap.type.value}: {preview}",
                    make_handler(cap)
                ))

            items.append(pystray.Menu.SEPARATOR)
            items.append(pystray.MenuItem("View All...", self._open_dashboard))

            return pystray.Menu(*items)

        except Exception as e:
            print(f"Recent captures error: {e}")
            return pystray.Menu(
                pystray.MenuItem("Error loading captures", None, enabled=False)
            )

    def _show_capture_detail(self, capture):
        """Show full capture in notification."""
        if self.tray_icon:
            try:
                self.tray_icon.notify(
                    f"{capture.type.value.title()}",
                    capture.content[:200]
                )
            except Exception as e:
                print(f"Notification error: {e}")

    def _create_menu(self):
        """Create tray menu with dynamic status header."""
        status = self._get_status_text()

        return pystray.Menu(
            pystray.MenuItem(status, None, enabled=False),  # Status header
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quick Capture", self.on_capture_clicked, default=True),
            pystray.MenuItem("Voice Capture", self.on_voice_clicked),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Focus Mode", pystray.Menu(
                pystray.MenuItem("Quick Focus (15 min)", lambda icon, item: self._start_focus(15)),
                pystray.MenuItem("Standard Focus (25 min)", lambda icon, item: self._start_focus(25)),
                pystray.MenuItem("Deep Work (45 min)", lambda icon, item: self._start_focus(45)),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem(self._get_focus_status(), None, enabled=False),
                pystray.MenuItem("End Session", self._end_focus, visible=self._focus_active()),
            )),
            pystray.MenuItem("Recent Captures", self._create_recent_menu()),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("View Morning Briefing", self._show_last_briefing,
                           enabled=self._last_briefing is not None),
            pystray.MenuItem("Open Dashboard", self._open_dashboard),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quit", self.on_quit_clicked),
        )

    def _refresh_menu(self):
        """Update tray menu to reflect current state."""
        if self.tray_icon:
            self.tray_icon.menu = self._create_menu()

    def _open_dashboard(self, icon=None, item=None):
        """Open web dashboard in browser."""
        webbrowser.open("http://localhost:8000/dashboard.html")

    # =========================================================================
    # SCHEDULER AND MORNING BRIEFING
    # =========================================================================

    def _run_scheduler(self):
        """Background scheduler for timed events."""
        while self.scheduler_running:
            try:
                self._check_morning_briefing()
            except Exception as e:
                print(f"Scheduler error: {e}")
            time.sleep(60)  # Check every minute

    def _check_morning_briefing(self):
        """Check if morning briefing should trigger."""
        try:
            if not self.morning_assistant:
                from modes.morning_briefing import MorningAssistant
                config = self._load_config()
                self.morning_assistant = MorningAssistant(config, user_name="Will")

            # Reset daily flag at midnight
            today = datetime.now().date()
            if self._last_briefing_date != today:
                self.briefing_shown_today = False
                self._last_briefing_date = today

            if self.morning_assistant.should_trigger() and not self.briefing_shown_today:
                self._show_morning_briefing()

        except Exception as e:
            print(f"Morning briefing check error: {e}")

    def _show_morning_briefing(self):
        """Generate and display morning briefing."""
        import asyncio

        def generate_and_show():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                briefing = loop.run_until_complete(
                    self.morning_assistant.generate_briefing()
                )
                self._last_briefing = briefing
                self.briefing_shown_today = True

                # Show tkinter notification window
                self._show_briefing_window(briefing)

            except Exception as e:
                print(f"Morning briefing error: {e}")
            finally:
                loop.close()

        threading.Thread(target=generate_and_show, daemon=True).start()

    def _show_briefing_window(self, briefing):
        """Display briefing in tkinter window."""
        if not HAS_TK:
            return

        def create_window():
            window = tk.Tk()
            window.title(f"Good Morning - {briefing.day_rating.title()} Day")
            window.geometry("400x320+50+50")
            window.configure(bg="#1a1a24")
            window.attributes("-topmost", True)

            # Header
            header = tk.Label(
                window, text=briefing.greeting,
                font=("Helvetica", 18, "bold"),
                fg="#6366f1", bg="#1a1a24"
            )
            header.pack(pady=(20, 10))

            # Weather
            weather_text = briefing.weather_summary or "Weather unavailable"
            weather_label = tk.Label(
                window, text=weather_text,
                font=("Helvetica", 12),
                fg="#ffffff", bg="#1a1a24",
                wraplength=360
            )
            weather_label.pack(pady=5)

            # Day rating
            rating_colors = {
                "light": "#22c55e",
                "normal": "#3b82f6",
                "busy": "#f59e0b",
                "packed": "#ef4444"
            }
            rating_color = rating_colors.get(briefing.day_rating, "#3b82f6")
            rating_label = tk.Label(
                window, text=f"{briefing.day_rating.title()} day",
                font=("Helvetica", 14, "bold"),
                fg=rating_color, bg="#1a1a24"
            )
            rating_label.pack(pady=5)

            # Items count
            from modes.morning_briefing import BriefingItemType
            calendar_count = len(briefing.get_by_type(BriefingItemType.CALENDAR))
            items_label = tk.Label(
                window, text=f"{calendar_count} events today • {len(briefing.items)} items",
                font=("Helvetica", 11),
                fg="#a0a0a0", bg="#1a1a24"
            )
            items_label.pack(pady=5)

            # Top 3 priority items
            priority_items = briefing.get_priority_items(3)
            for item in priority_items:
                item_label = tk.Label(
                    window, text=f"• {item.title}: {item.content[:50]}",
                    font=("Helvetica", 10),
                    fg="#d0d0d0", bg="#1a1a24",
                    wraplength=360, justify="left"
                )
                item_label.pack(pady=2, anchor="w", padx=20)

            # Buttons frame
            btn_frame = tk.Frame(window, bg="#1a1a24")
            btn_frame.pack(pady=20)

            # Dashboard button
            dashboard_btn = tk.Button(
                btn_frame, text="Open Dashboard",
                command=lambda: [webbrowser.open("http://localhost:8000/dashboard.html"), window.destroy()],
                bg="#3b82f6", fg="#ffffff",
                font=("Helvetica", 11), padx=15
            )
            dashboard_btn.pack(side=tk.LEFT, padx=5)

            # Dismiss button
            dismiss_btn = tk.Button(
                btn_frame, text="Dismiss",
                command=window.destroy,
                bg="#6366f1", fg="#ffffff",
                font=("Helvetica", 11), padx=15
            )
            dismiss_btn.pack(side=tk.LEFT, padx=5)

            # Auto-close after 30 seconds
            window.after(30000, window.destroy)
            window.mainloop()

        threading.Thread(target=create_window, daemon=True).start()

    def _show_last_briefing(self, icon=None, item=None):
        """Show last morning briefing."""
        if self._last_briefing:
            threading.Thread(
                target=lambda: self._show_briefing_window(self._last_briefing),
                daemon=True
            ).start()

    # =========================================================================
    # FOCUS MODE
    # =========================================================================

    def _start_focus(self, minutes: int):
        """Start focus session."""
        def start():
            import asyncio

            try:
                from modes.focus_mode import FocusMode

                if not self.focus_mode:
                    self.focus_mode = FocusMode(self._load_config())

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(
                        self.focus_mode.start_session("Desktop Focus", focus_min=minutes)
                    )
                    self._refresh_menu()

                    # Show notification
                    if self.tray_icon:
                        try:
                            self.tray_icon.notify(
                                "Focus Started",
                                f"{minutes} minutes of deep work. Good luck!"
                            )
                        except:
                            pass
                finally:
                    loop.close()

            except Exception as e:
                print(f"Focus start error: {e}")

        threading.Thread(target=start, daemon=True).start()

    def _end_focus(self, icon=None, item=None):
        """End focus session."""
        def end():
            import asyncio

            if self.focus_mode:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.focus_mode.end_session())
                    self._refresh_menu()

                    if self.tray_icon:
                        try:
                            self.tray_icon.notify("Focus Complete", "Great work!")
                        except:
                            pass
                except Exception as e:
                    print(f"Focus end error: {e}")
                finally:
                    loop.close()

        threading.Thread(target=end, daemon=True).start()

    # =========================================================================
    # QUICK CAPTURE WITH AUTO-SAVE DRAFTS
    # =========================================================================

    def _save_draft(self, text: str, capture_type: str):
        """Save capture draft to disk."""
        try:
            DRAFTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            DRAFTS_FILE.write_text(json.dumps({
                "text": text,
                "type": capture_type,
                "timestamp": datetime.now().isoformat()
            }))
        except Exception as e:
            print(f"Draft save error: {e}")

    def _load_draft(self) -> Optional[dict]:
        """Load saved draft if exists."""
        if DRAFTS_FILE.exists():
            try:
                return json.loads(DRAFTS_FILE.read_text())
            except:
                pass
        return None

    def _clear_draft(self):
        """Clear saved draft after successful save."""
        try:
            if DRAFTS_FILE.exists():
                DRAFTS_FILE.unlink()
        except:
            pass

    def on_capture_clicked(self, icon=None, item=None):
        """Handle capture menu click."""
        self.show_capture_window()

    def show_capture_window(self):
        """Show the quick capture popup window with auto-save drafts."""
        if not HAS_TK:
            print("Tkinter not available")
            return

        def create_window():
            window = tk.Tk()
            window.title("WHAM Quick Capture")
            window.geometry("400x220")
            window.attributes("-topmost", True)
            window.configure(bg="#1a1a24")

            # Title
            title = tk.Label(
                window,
                text="Quick Capture",
                font=("Helvetica", 16, "bold"),
                fg="#ffffff",
                bg="#1a1a24"
            )
            title.pack(pady=(15, 10))

            # Text entry
            text_var = tk.StringVar()
            entry = tk.Entry(
                window,
                textvariable=text_var,
                font=("Helvetica", 14),
                width=40,
                bg="#2a2a3a",
                fg="#ffffff",
                insertbackground="#ffffff"
            )
            entry.pack(pady=10, padx=20)
            entry.focus_set()

            # Type selector frame
            frame = tk.Frame(window, bg="#1a1a24")
            frame.pack(pady=10)

            type_var = tk.StringVar(value="note")
            types = [("Note", "note"), ("Todo", "todo"), ("Idea", "idea"), ("Reminder", "reminder")]

            for text, value in types:
                rb = tk.Radiobutton(
                    frame,
                    text=text,
                    variable=type_var,
                    value=value,
                    bg="#1a1a24",
                    fg="#a0a0b0",
                    selectcolor="#2a2a3a",
                    activebackground="#1a1a24"
                )
                rb.pack(side=tk.LEFT, padx=5)

            # Draft status label
            draft_label = tk.Label(
                window,
                text="",
                font=("Helvetica", 9),
                fg="#6366f1",
                bg="#1a1a24"
            )
            draft_label.pack()

            # Load existing draft
            draft = self._load_draft()
            if draft:
                text_var.set(draft.get("text", ""))
                type_var.set(draft.get("type", "note"))
                draft_label.config(text="Draft recovered")

            # Auto-save on text change
            def on_text_change(*args):
                text = text_var.get().strip()
                if text:
                    self._save_draft(text, type_var.get())
                    draft_label.config(text="Draft saved")

            text_var.trace_add("write", on_text_change)

            def submit():
                text = text_var.get().strip()
                if text:
                    self.save_capture(text, type_var.get())
                    self._clear_draft()
                    window.destroy()

            def on_enter(event):
                submit()

            entry.bind("<Return>", on_enter)

            # Submit button
            btn = tk.Button(
                window,
                text="Save (Enter)",
                command=submit,
                bg="#6366f1",
                fg="#ffffff",
                font=("Helvetica", 12),
                padx=20,
                pady=5
            )
            btn.pack(pady=10)

            # Escape to close
            window.bind("<Escape>", lambda e: window.destroy())

            # Center on screen
            window.update_idletasks()
            width = window.winfo_width()
            height = window.winfo_height()
            x = (window.winfo_screenwidth() // 2) - (width // 2)
            y = (window.winfo_screenheight() // 2) - (height // 2)
            window.geometry(f"+{x}+{y}")

            window.mainloop()

        thread = threading.Thread(target=create_window)
        thread.daemon = True
        thread.start()

    def save_capture(self, text: str, capture_type: str = "note"):
        """Save capture to API."""
        if not HAS_REQUESTS:
            print(f"Would capture: [{capture_type}] {text}")
            self._save_local(text, capture_type)
            return

        try:
            response = requests.post(
                f"{self.API_BASE}/captures",
                json={
                    "text": text,
                    "type": capture_type,
                    "priority": "normal",
                    "tags": []
                },
                timeout=5
            )

            if response.status_code == 200:
                print(f"✓ Captured: {text[:50]}...")
            else:
                print(f"✗ API error: {response.status_code}")
                self._save_local(text, capture_type)

        except requests.exceptions.ConnectionError:
            print("✗ API not available, saving locally")
            self._save_local(text, capture_type)
        except Exception as e:
            print(f"✗ Error: {e}")
            self._save_local(text, capture_type)

    def _save_local(self, text: str, capture_type: str):
        """Save capture locally when API is unavailable."""
        local_file = PHONE_CLIENT_PATH / "captures" / "pending_sync.json"
        local_file.parent.mkdir(parents=True, exist_ok=True)

        pending = []
        if local_file.exists():
            try:
                with open(local_file) as f:
                    pending = json.load(f)
            except:
                pass

        pending.append({
            "text": text,
            "type": capture_type,
            "created_at": datetime.now().isoformat(),
            "synced": False
        })

        with open(local_file, "w") as f:
            json.dump(pending, f, indent=2)

        print(f"✓ Saved locally ({len(pending)} pending)")

    # =========================================================================
    # VOICE RECORDING WITH WAVEFORM VISUALIZATION
    # =========================================================================

    def on_voice_clicked(self, icon=None, item=None):
        """Handle voice capture click."""
        if not HAS_AUDIO:
            print("Audio recording not available (pyaudio not installed)")
            return

        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            self.show_recording_window()

    def show_recording_window(self):
        """Show recording window with waveform visualization."""
        if not HAS_TK:
            return

        def create_window():
            self.recording_window = tk.Tk()
            self.recording_window.title("WHAM Voice Capture")
            self.recording_window.geometry("420x180")
            self.recording_window.configure(bg="#1a1a24")
            self.recording_window.attributes("-topmost", True)

            # Status label
            self.recording_label = tk.Label(
                self.recording_window,
                text="🎤 Recording...",
                font=("Helvetica", 14, "bold"),
                fg="#ff6b6b", bg="#1a1a24"
            )
            self.recording_label.pack(pady=10)

            # Waveform canvas
            self.waveform_canvas = tk.Canvas(
                self.recording_window,
                width=380, height=80,
                bg="#2a2a3a", highlightthickness=0
            )
            self.waveform_canvas.pack(pady=10)

            # Stop button
            stop_btn = tk.Button(
                self.recording_window,
                text="Stop Recording",
                command=self._stop_recording_from_window,
                bg="#ff6b6b", fg="#ffffff",
                font=("Helvetica", 11),
                padx=15
            )
            stop_btn.pack(pady=10)

            # Escape to stop
            self.recording_window.bind("<Escape>", lambda e: self._stop_recording_from_window())

            # Start waveform updates
            self._update_waveform()
            self.recording_window.mainloop()

        threading.Thread(target=create_window, daemon=True).start()

    def _update_waveform(self):
        """Update waveform visualization from audio buffer."""
        if not hasattr(self, 'recording_window') or not self.recording_window:
            return
        if not self.is_recording:
            return
        if not self.waveform_canvas:
            return

        try:
            canvas = self.waveform_canvas
            canvas.delete("all")

            if self.audio_frames and len(self.audio_frames) > 0:
                # Get most recent audio frame
                frame = self.audio_frames[-1]
                samples = struct.unpack(f'{len(frame)//2}h', frame)

                width, height = 380, 80
                center_y = height // 2
                num_bars = 50
                samples_per_bar = max(1, len(samples) // num_bars)
                bar_width = width // num_bars

                for i in range(min(num_bars, len(samples) // samples_per_bar)):
                    start_idx = i * samples_per_bar
                    end_idx = min(start_idx + samples_per_bar, len(samples))
                    avg = sum(abs(s) for s in samples[start_idx:end_idx]) / max(1, end_idx - start_idx)
                    bar_height = min(int((avg / 32768) * height * 1.5), height - 4)

                    x = i * bar_width + 4
                    y1 = center_y - bar_height // 2
                    y2 = center_y + bar_height // 2

                    # Color based on amplitude (purple to pink)
                    intensity = min(255, int((avg / 32768) * 255 * 2))
                    r = min(255, 99 + intensity // 3)
                    g = 102
                    b = max(100, 241 - intensity // 2)
                    color = f"#{r:02x}{g:02x}{b:02x}"

                    canvas.create_rectangle(x, y1, x + bar_width - 2, y2, fill=color, outline="")
            else:
                # Draw flat line when no audio
                canvas.create_line(0, 40, 380, 40, fill="#6366f1", width=2)

            # Schedule next update at ~20fps
            if self.is_recording and self.recording_window:
                self.recording_window.after(50, self._update_waveform)

        except tk.TclError:
            # Window was destroyed
            pass
        except Exception as e:
            print(f"Waveform update error: {e}")

    def _stop_recording_from_window(self):
        """Stop recording and close window."""
        self.stop_recording()
        if hasattr(self, 'recording_window') and self.recording_window:
            try:
                self.recording_window.destroy()
            except:
                pass
            self.recording_window = None
            self.waveform_canvas = None

    def start_recording(self):
        """Start audio recording."""
        if not HAS_AUDIO:
            return

        print("🎤 Recording... (press hotkey again to stop)")
        self.is_recording = True
        self.audio_frames = []

        self.audio_interface = pyaudio.PyAudio()
        self.audio_stream = self.audio_interface.open(
            format=self.AUDIO_FORMAT,
            channels=1,
            rate=self.AUDIO_RATE,
            input=True,
            frames_per_buffer=self.AUDIO_CHUNK
        )

        # Record in background
        def record():
            while self.is_recording:
                try:
                    data = self.audio_stream.read(self.AUDIO_CHUNK)
                    self.audio_frames.append(data)
                except Exception as e:
                    print(f"Recording error: {e}")
                    break

        thread = threading.Thread(target=record)
        thread.daemon = True
        thread.start()

    def stop_recording(self):
        """Stop recording and transcribe."""
        if not self.is_recording:
            return

        print("🛑 Stopping recording...")
        self.is_recording = False

        if self.audio_stream:
            try:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            except:
                pass
        if self.audio_interface:
            try:
                self.audio_interface.terminate()
            except:
                pass

        if not self.audio_frames:
            print("No audio recorded")
            return

        # Save to temp file
        temp_file = Path("/tmp/wham_voice_capture.wav")
        with wave.open(str(temp_file), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(self.AUDIO_RATE)
            wf.writeframes(b"".join(self.audio_frames))

        print(f"📝 Transcribing ({len(self.audio_frames)} frames)...")

        # Transcribe with Whisper
        if HAS_WHISPER:
            try:
                if self.whisper_model is None:
                    print("Loading Whisper model (first time)...")
                    self.whisper_model = whisper.load_model("base")

                result = self.whisper_model.transcribe(str(temp_file))
                text = result["text"].strip()

                if text:
                    print(f"Transcribed: {text}")
                    self.save_capture(text, "voice")
                else:
                    print("No speech detected")

            except Exception as e:
                print(f"Transcription error: {e}")
        else:
            print("Whisper not available - audio saved but not transcribed")

        # Cleanup
        self.audio_frames = []

    # =========================================================================
    # HOTKEY AND APPLICATION LIFECYCLE
    # =========================================================================

    def setup_hotkey(self):
        """Set up global hotkey."""
        if not HAS_KEYBOARD:
            print("Keyboard hotkeys not available")
            return

        try:
            keyboard.add_hotkey(self.HOTKEY, self.on_hotkey)
            print(f"✓ Global hotkey registered: {self.HOTKEY}")
        except Exception as e:
            print(f"✗ Failed to register hotkey: {e}")

    def on_hotkey(self):
        """Handle hotkey press."""
        if self.is_recording:
            self.stop_recording()
        else:
            self.show_capture_window()

    def start(self):
        """Start the application."""
        self.running = True

        print("\n" + "=" * 50)
        print("   WHAM Quick Capture - Enhanced")
        print("=" * 50)
        print(f"\nHotkey: {self.HOTKEY}")
        print("Right-click tray icon for menu")
        print("\nFeatures:")
        print("  • Morning briefing notifications")
        print("  • Focus mode (15/25/45 min)")
        print("  • Voice recording with waveform")
        print("  • Auto-save drafts")
        print("  • Recent captures menu")
        print("\nPress Ctrl+C to quit\n")

        # Setup hotkey
        self.setup_hotkey()

        # Start scheduler thread
        self.scheduler_running = True
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        print("✓ Background scheduler started")

        # Create tray icon
        if HAS_TRAY:
            self.tray_icon = pystray.Icon(
                "WHAM",
                self.create_tray_icon(),
                "WHAM Quick Capture",
                self._create_menu()
            )

            # Run tray icon (blocking)
            self.tray_icon.run()
        else:
            # No tray - just wait
            print("Running without system tray (pystray not installed)")
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass

    def stop(self):
        """Stop the application."""
        print("\nShutting down...")
        self.running = False
        self.scheduler_running = False

        if HAS_KEYBOARD:
            try:
                keyboard.unhook_all()
            except:
                pass

        if self.tray_icon:
            self.tray_icon.stop()

    def on_quit_clicked(self, icon=None, item=None):
        """Handle quit click."""
        self.stop()


def main():
    """Main entry point."""
    app = QuickCaptureApp()
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
