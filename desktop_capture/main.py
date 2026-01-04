"""
WHAM Desktop Quick Capture - System Tray Application.
Global hotkey for quick capture without opening browser.
"""
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add phone_client to path
PHONE_CLIENT_PATH = Path(__file__).parent.parent / "phone_client"
sys.path.insert(0, str(PHONE_CLIENT_PATH))

# GUI imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
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


class QuickCaptureApp:
    """
    Desktop Quick Capture Application.

    Features:
    - System tray icon
    - Global hotkey (Ctrl+Shift+W / Cmd+Shift+W)
    - Popup text input for quick capture
    - Voice recording with Whisper transcription
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
        self.whisper_model = None

        # Recording state
        self.is_recording = False
        self.audio_frames = []
        self.audio_stream = None
        self.audio_interface = None

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

        if missing:
            print("\n⚠️  Missing optional dependencies:")
            for dep in missing:
                print(f"   - {dep}")
            print("\nBasic functionality available. Install missing for full features.\n")

    def create_tray_icon(self) -> Image.Image:
        """Create the system tray icon."""
        # Create a simple icon
        size = 64
        image = Image.new('RGBA', (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw a stylized "W" for WHAM
        draw.ellipse([4, 4, size-4, size-4], fill=(99, 102, 241))
        draw.text((size//2 - 12, size//2 - 14), "W", fill="white")

        return image

    def on_capture_clicked(self, icon=None, item=None):
        """Handle capture menu click."""
        self.show_capture_window()

    def on_voice_clicked(self, icon=None, item=None):
        """Handle voice capture click."""
        if not HAS_AUDIO:
            print("Audio recording not available (pyaudio not installed)")
            return

        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def on_quit_clicked(self, icon=None, item=None):
        """Handle quit click."""
        self.stop()

    def show_capture_window(self):
        """Show the quick capture popup window."""
        if not HAS_TK:
            print("Tkinter not available")
            return

        # Create window in new thread to avoid blocking
        def create_window():
            window = tk.Tk()
            window.title("WHAM Quick Capture")
            window.geometry("400x200")
            window.attributes("-topmost", True)

            # Dark theme
            window.configure(bg="#1a1a24")

            # Title
            title = tk.Label(
                window,
                text="Quick Capture",
                font=("Helvetica", 16, "bold"),
                fg="#ffffff",
                bg="#1a1a24"
            )
            title.pack(pady=(20, 10))

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

            # Type selector
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

            def submit():
                text = text_var.get().strip()
                if text:
                    self.save_capture(text, type_var.get())
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

        # Run in thread
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
            with open(local_file) as f:
                pending = json.load(f)

        pending.append({
            "text": text,
            "type": capture_type,
            "created_at": datetime.now().isoformat(),
            "synced": False
        })

        with open(local_file, "w") as f:
            json.dump(pending, f, indent=2)

        print(f"✓ Saved locally ({len(pending)} pending)")

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
            self.audio_stream.stop_stream()
            self.audio_stream.close()
        if self.audio_interface:
            self.audio_interface.terminate()

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
        print("   WHAM Quick Capture")
        print("=" * 50)
        print(f"\nHotkey: {self.HOTKEY}")
        print("Right-click tray icon for menu")
        print("\nPress Ctrl+C to quit\n")

        # Setup hotkey
        self.setup_hotkey()

        # Create tray icon
        if HAS_TRAY:
            menu = pystray.Menu(
                pystray.MenuItem("Quick Capture", self.on_capture_clicked, default=True),
                pystray.MenuItem("Voice Capture", self.on_voice_clicked),
                pystray.Menu.SEPARATOR,
                pystray.MenuItem("Quit", self.on_quit_clicked)
            )

            self.tray_icon = pystray.Icon(
                "WHAM",
                self.create_tray_icon(),
                "WHAM Quick Capture",
                menu
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

        if HAS_KEYBOARD:
            try:
                keyboard.unhook_all()
            except:
                pass

        if self.tray_icon:
            self.tray_icon.stop()


def main():
    """Main entry point."""
    app = QuickCaptureApp()
    try:
        app.start()
    except KeyboardInterrupt:
        app.stop()


if __name__ == "__main__":
    main()
