# DogVetReviewer2.1.py
# Clean, single-file Tkinter app with VLC playback, gallery tabs (Inbox/Allow/Deny),
# refresh, Faster-Whisper transcription (with grace period) and optional openai-whisper fallback,
# profanity highlighting, folder gating, portable VLC/FFmpeg, robust threading/shutdown,
# a GUI-only progress bar next to the "Transcribing… XX%" label with ETA and speed colors,
# and named fonts for easy resizing of the progress/status texts.

import csv, shutil, subprocess, threading, os, math, tempfile, sys, traceback, re, platform, json, hashlib
from datetime import datetime
from time import time  # ETA timing
from pathlib import Path
from typing import Optional, List, Tuple
from tkinter import Tk, Frame, Label, StringVar, HORIZONTAL, messagebox, Scale, Button, Canvas, Scrollbar, Menu
from tkinter.filedialog import askdirectory
from tkinter.scrolledtext import ScrolledText
from tkinter.simpledialog import askstring
from tkinter import ttk  # GUI progress bar + styles
from tkinter.font import Font  # named fonts for easy resizing

# --- Hide spawned console windows on Windows (for ffmpeg, etc.) ---
if os.name == "nt":
    _CREATE_NO_WINDOW = 0x08000000
    _si = subprocess.STARTUPINFO()
    _si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _NO_CONSOLE = {"startupinfo": _si, "creationflags": _CREATE_NO_WINDOW}
else:
    _NO_CONSOLE = {}


# Mute library chatter in console
import logging
for _n in ("faster_whisper", "ctranslate2", "torch"):
    logging.getLogger(_n).setLevel(logging.ERROR)

# --- early crash logging & OpenMP fix ---
try:
    import faulthandler
    DATA_ROOT_EARLY = Path(os.getenv("LOCALAPPDATA") or os.path.expanduser("~")) / "DogVet"
    (DATA_ROOT_EARLY / "logs").mkdir(parents=True, exist_ok=True)
    _fh = open(DATA_ROOT_EARLY / "logs" / "crash_faulthandler.log", "a", encoding="utf-8")
    faulthandler.enable(_fh)
except Exception:
    pass

def _install_global_excepthook():
    from datetime import datetime as _dt
    LOGS_DIR_ = Path(os.getenv("LOCALAPPDATA") or os.path.expanduser("~")) / "DogVet" / "logs"
    LOGS_DIR_.mkdir(parents=True, exist_ok=True)
    def _global_excepthook(exctype, value, tb):
        try:
            with open(LOGS_DIR_ / "crash_unhandled.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{_dt.now():%Y-%m-%d %H:%M:%S}] Unhandled exception\n")
                traceback.print_exception(exctype, value, tb, file=f)
        except Exception:
            pass
        sys.__excepthook__(exctype, value, tb)
    sys.excepthook = _global_excepthook
_install_global_excepthook()

# Avoid OpenMP clash (torch/opencv) in frozen apps
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ---------- Optional dependencies for thumbnails ----------
try:
    import cv2
except Exception:
    cv2 = None
try:
    from PIL import Image, ImageTk
except Exception:
    Image = ImageTk = None

# ---------- Speech backends ----------
WhisperModel = None
try:
    from faster_whisper import WhisperModel  # preferred
except Exception:
    WhisperModel = None

try:
    import whisper as whisper_lib  # openai-whisper (fallback)
except Exception:
    whisper_lib = None

# ---------- Profanity ----------
try:
    from better_profanity import profanity
except Exception:
    profanity = None

# ===================== CONSTANTS =====================
APP_NAME = "DogVet Video Reviewer"
APP_VER  = "2.3"

# Whisper (and Faster-Whisper) model variants:
#
# Name    | Approx Size | Speed    | Accuracy         | Typical Use
# --------+-------------+----------+------------------+----------------------------
# tiny    | ~39 MB      | Fastest  | Least accurate   | Quick/offline checks
# base    | ~74 MB      | Fast     | Good balance     | Common for desktop
# small   | ~244 MB     | Medium   | Better accuracy  | When you have more CPU/GPU
# medium  | ~769 MB     | Slower   | Very accurate    | Production servers
# large   | ~1.5 GB     | Slowest  | Highest accuracy | GPU-only scenarios

# Model name used by both backends; keep small-ish for portability
MODEL_NAME = "base"  # multilingual (supports English + Spanish)

# Language mode:
#   "en"   → English only
#   "es"   → Spanish only
#   "auto" → Automatic detection (recommended for English + Spanish)
LANGUAGE = {"value": "auto"}  # mutable holder so closures can modify

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac"}
PLAYER_W, PLAYER_H = 400, 250
GALLERY_W_FALLBACK = 660
THUMB_LABEL_H = 20
VISIBLE_ROWS = 3
VISIBLE_COLS = 4
GALLERY_VPAD = 6
GALLERY_HPAD = 6
TRANSCRIPT_FONT = ("Segoe UI", 12)
SAFE_MIN_CHUNK_SEC = 0.2
SAFE_MIN_AUDIO_BYTES = 2048
CHUNK_SEC = 30.0
NO_SPEECH_GRACE_SEC = 6.0

# ===================== PATHS / LOG =====================
APP_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)).resolve()
DATA_ROOT = Path(os.getenv("LOCALAPPDATA") or os.path.expanduser("~")) / "DogVet"
for sub in ("All_media", "allow", "deny", "transcripts", "logs", "models"):
    (DATA_ROOT / sub).mkdir(parents=True, exist_ok=True)

INBOX_DIR       = (DATA_ROOT / "All_media")
ALLOW_DIR       = (DATA_ROOT / "allow")
DENY_DIR        = (DATA_ROOT / "deny")
TRANSCRIPTS_DIR = (DATA_ROOT / "transcripts")
LOGS_DIR        = (DATA_ROOT / "logs")
MODELS_DIR      = (DATA_ROOT / "models")
CONFIG_JSON     = (DATA_ROOT / "config.json")   # delete this file to reset password
DECISIONS_CSV   = (LOGS_DIR / "decisions.csv")
BADWORDS_TXT    = (APP_ROOT / "badwords.txt")  # optional

def _log_dir(p: Path):
    try: p.mkdir(parents=True, exist_ok=True)
    except Exception: pass

def log_info(msg: str):
    try:
        _log_dir(LOGS_DIR)
        with open(LOGS_DIR / "app.log", "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}\n")
    except Exception:
        pass

def log_exc(prefix=""):
    try:
        _log_dir(LOGS_DIR)
        with open(LOGS_DIR / "app.log", "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now():%Y-%m-%d %H:%M:%S}] {prefix}\n")
            traceback.print_exc(file=f)
    except Exception:
        pass

# ===================== VLC (DLLs) =====================
if sys.platform.startswith("win"):
    VLC_DIR_PORTABLE = APP_ROOT / "vlc_runtime"
    if VLC_DIR_PORTABLE.is_dir():
        try:
            os.add_dll_directory(str(VLC_DIR_PORTABLE))
        except Exception:
            os.environ["PATH"] = str(VLC_DIR_PORTABLE) + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("VLC_PLUGIN_PATH", str(VLC_DIR_PORTABLE / "plugins"))
    else:
        VLC_DIR = r"C:\Program Files\VideoLAN\VLC"
        try:
            os.add_dll_directory(VLC_DIR)
        except Exception:
            os.environ["PATH"] = VLC_DIR + os.pathsep + os.environ.get("PATH", "")
        os.environ.setdefault("VLC_PLUGIN_PATH", os.path.join(VLC_DIR, "plugins"))

try:
    import vlc
except Exception:
    vlc = None
    log_exc("import vlc failed")

# ===================== FFmpeg PATHING =====================
FFMPEG_DIR = APP_ROOT / "ffmpeg" / "bin"  # if bundled
if FFMPEG_DIR.is_dir():
    os.environ["PATH"] = str(FFMPEG_DIR) + os.pathsep + os.environ.get("PATH", "")
COMMON_FFMPEG = Path(r"C:\ffmpeg\bin")
if COMMON_FFMPEG.is_dir():
    os.environ["PATH"] = str(COMMON_FFMPEG) + os.pathsep + os.environ.get("PATH", "")

def _ffbin(which: str) -> Optional[str]:
    return shutil.which(which)

# ===================== PERSISTENCE / PASSWORD =====================
_config_cache = {
    "inbox": str(INBOX_DIR),
    "allow": str(ALLOW_DIR),
    "deny":  str(DENY_DIR),
    "folders_password_hash": None,
    "language": LANGUAGE["value"],
}

def _ensure_dir(p: Path) -> Path:
    try: p.mkdir(parents=True, exist_ok=True)
    except Exception: pass
    return p

def _hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode("utf-8")).hexdigest()

def _save_config():
    try:
        data = {
            "inbox": str(INBOX_DIR),
            "allow": str(ALLOW_DIR),
            "deny":  str(DENY_DIR),
            "folders_password_hash": _config_cache.get("folders_password_hash"),
            "language": LANGUAGE["value"],
        }
        CONFIG_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        log_exc("save_config")

def _load_config():
    global INBOX_DIR, ALLOW_DIR, DENY_DIR, _config_cache
    try:
        if CONFIG_JSON.exists():
            data = json.loads(CONFIG_JSON.read_text(encoding="utf-8"))
            if "inbox" in data: INBOX_DIR = Path(data["inbox"])
            if "allow" in data: ALLOW_DIR = Path(data["allow"])
            if "deny"  in data: DENY_DIR  = Path(data["deny"])
            _config_cache["folders_password_hash"] = data.get("folders_password_hash")
            if "language" in data:
                LANGUAGE["value"] = data["language"]
            for d in (INBOX_DIR, ALLOW_DIR, DENY_DIR):
                d.mkdir(parents=True, exist_ok=True)
    except Exception:
        log_exc("load_config")

def require_folders_password() -> bool:
    try:
        stored = _config_cache.get("folders_password_hash")
        if not stored:
            resp = messagebox.askyesno("Set Password", "No folders password set.\nCreate one now?")
            if not resp: return False
            while True:
                pw1 = askstring("Create Password", "Enter new password:", show="*")
                if pw1 is None: return False
                pw2 = askstring("Confirm Password", "Re-enter password:", show="*")
                if pw2 is None: return False
                if pw1 != pw2:
                    messagebox.showerror("Mismatch", "Passwords do not match.")
                    continue
                if len(pw1) < 4:
                    messagebox.showerror("Too short", "Use at least 4 characters.")
                    continue
                _config_cache["folders_password_hash"] = _hash_password(pw1)
                _save_config()
                messagebox.showinfo("Password Set", "Folders password created.")
                return True
        else:
            pw = askstring("Password Required", "Enter folders password:", show="*")
            if pw is None: return False
            if _hash_password(pw) != stored:
                messagebox.showerror("Access denied", "Incorrect password.")
                return False
            return True
    except Exception:
        log_exc("require_folders_password")
        return False

def set_or_change_folders_password():
    try:
        stored = _config_cache.get("folders_password_hash")
        if stored:
            old = askstring("Change Password", "Enter current password:", show="*")
            if old is None: return
            if _hash_password(old) != stored:
                messagebox.showerror("Access denied", "Incorrect current password.")
                return
        while True:
            pw1 = askstring("New Password", "Enter new password:", show="*")
            if pw1 is None: return
            pw2 = askstring("Confirm Password", "Re-enter new password:", show="*")
            if pw2 is None: return
            if pw1 != pw2:
                messagebox.showerror("Mismatch", "Passwords do not match.")
                continue
            if len(pw1) < 4:
                messagebox.showerror("Too short", "Use at least 4 characters.")
                continue
            _config_cache["folders_password_hash"] = _hash_password(pw1)
            _save_config()
            messagebox.showinfo("Password Updated", "Folders password updated.")
            return
    except Exception:
        log_exc("set_or_change_folders_password")

# ===================== DATA / GALLERY =====================
def ensure_csv():
    if not DECISIONS_CSV.exists():
        with open(DECISIONS_CSV, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["timestamp", "filename", "decision"])

def append_decision(filename, decision):
    ensure_csv()
    with open(DECISIONS_CSV, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), filename, decision])

def list_from_dir(dir_path: Path) -> List[Path]:
    try:
        items = [p for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in (VIDEO_EXTS | AUDIO_EXTS)]
        items.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return items
    except Exception:
        return []

def human_tab_name(key: str) -> str:
    return {"inbox": "Inbox", "allow": "Allow", "deny": "Deny"}.get(key, key.title())

def count_media(dir_path: Path) -> int:
    try:
        return sum(1 for p in dir_path.iterdir() if p.is_file() and p.suffix.lower() in (VIDEO_EXTS | AUDIO_EXTS))
    except Exception:
        return 0

def get_tab_counts():
    return {
        "inbox": count_media(INBOX_DIR),
        "allow": count_media(ALLOW_DIR),
        "deny":  count_media(DENY_DIR),
    }

# ===================== TRANSCRIPTION =====================
_fw_model = None
_ws_model = None

def _load_faster_whisper_once(status_set):
    """Load faster-whisper model once (preferred if installed)."""
    global _fw_model
    if _fw_model is not None:
        return
    if WhisperModel is None:
        return
    status_set("Loading Faster-Whisper model…")
    try:
        _fw_model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        status_set("Model ready (Faster-Whisper)")
    except Exception as e:
        _fw_model = None
        log_exc("load_faster_whisper_once")
        status_set(f"Faster-Whisper load failed: {e}")

def _load_whisper_once(status_set):
    """Load openai-whisper model once (fallback)."""
    global _ws_model
    if _ws_model is not None:
        return
    if whisper_lib is None:
        status_set("Whisper fallback not installed")
        return
    status_set("Loading Whisper (fallback)…")
    try:
        model_name = MODEL_NAME
        if LANGUAGE["value"] == "en" and not MODEL_NAME.endswith(".en"):
            model_name = f"{MODEL_NAME}.en"
        _ws_model = whisper_lib.load_model(model_name, device="cpu", download_root=str(MODELS_DIR))
        status_set("Model ready (Whisper fallback)")
    except Exception as e:
        _ws_model = None
        log_exc("load_whisper_once")
        status_set(f"Whisper load failed: {e}")

def _wav_duration_seconds(wav_path: Path) -> float:
    import wave, contextlib
    try:
        with contextlib.closing(wave.open(str(wav_path), 'rb')) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 16000
            return frames / float(rate) if rate > 0 else 0.0
    except Exception:
        log_exc("wav_duration_seconds")
        return 0.0

def _safe_transcribe_seg(seg_path: Path) -> str:
    """Transcribe one segment with whichever backend is available."""
    lang = LANGUAGE["value"]
    text = ""
    # Try faster-whisper first
    if _fw_model is not None:
        try:
            fw_language = None if lang == "auto" else lang
            segments, info = _fw_model.transcribe(str(seg_path), language=fw_language, vad_filter=False)
            chunks = [s.text for s in segments]
            text = " ".join(chunks).strip()
            if text:
                return text
        except Exception:
            log_exc(f"faster_whisper transcribe chunk {seg_path.name}")

    # Fallback to openai-whisper
    if _ws_model is not None:
        try:
            kw = {"verbose": False, "fp16": False}
            if lang != "auto":
                kw["language"] = lang
            out = _ws_model.transcribe(str(seg_path), **kw)
            text = (out.get("text") or "").strip()
            return text
        except Exception:
            log_exc(f"whisper transcribe chunk {seg_path.name}")

    return text

def transcribe_file(src_path: Path, ui_status, ui_pct, ui_transcript, write_output=True):
    """FFmpeg → 16k WAV → chunk → (faster-)whisper."""
    ffmpeg = _ffbin("ffmpeg")
    if not ffmpeg:
        ui_status("Transcription error: FFmpeg not found")
        ui_pct("Transcription error")
        return

    def run():
        start_ts = datetime.now().timestamp()

        # Prefer faster-whisper if present, otherwise whisper
        if WhisperModel is not None:
            _load_faster_whisper_once(ui_status)
        if _fw_model is None:
            _load_whisper_once(ui_status)
        if (_fw_model is None) and (_ws_model is None):
            ui_status("No transcription backend available")
            ui_pct("Transcription error")
            return

        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio16k.wav"
            try:
                proc = subprocess.run(
                    [ffmpeg, "-y", "-i", str(src_path), "-vn", "-ac", "1", "-ar", "16000",
                     "-acodec", "pcm_s16le", str(wav)],
                    check=False, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    **_NO_CONSOLE
                )

                if proc.returncode != 0:
                    try:
                        (LOGS_DIR / "ffmpeg_decode.log").write_text(proc.stderr.decode(errors="ignore"), encoding="utf-8")
                    except Exception:
                        pass
                    ui_status("Transcription error: ffmpeg decode failed")
                    ui_pct("Transcription error")
                    return
            except Exception:
                log_exc("ffmpeg reencode_to_wav16k")
                ui_status("Transcription error: ffmpeg decode failed")
                ui_pct("Transcription error")
                return

            total = _wav_duration_seconds(wav)
            pieces: List[str] = []

            def _update_transcript_now():
                cur = "\n".join(pieces).strip()
                ui_transcript(cur)

            # Unknown duration — single shot with grace
            if total <= 0.0:
                ui_status("Transcribing…")
                txt = _safe_transcribe_seg(wav)
                if txt:
                    pieces.append(txt)
                    if write_output:
                        (TRANSCRIPTS_DIR / f"{src_path.stem}.txt").write_text(txt, encoding="utf-8")
                    _update_transcript_now()
                    ui_pct("Transcription complete.")
                    ui_status("Transcription done")
                else:
                    elapsed = datetime.now().timestamp() - start_ts
                    if elapsed < NO_SPEECH_GRACE_SEC:
                        ui_pct("Transcription complete.")
                        ui_status("Transcription done")
                        _update_transcript_now()
                    else:
                        ui_pct("No speech")
                        ui_status("Transcription done")
                        _update_transcript_now()
                return

            # Chunked path
            num_chunks = int(math.ceil(total / CHUNK_SEC))
            for i in range(num_chunks):
                start = i * CHUNK_SEC
                length = min(CHUNK_SEC, max(0.0, total - start))
                if length < SAFE_MIN_CHUNK_SEC:
                    continue
                seg = Path(td) / f"seg_{i:03}.wav"
                try:
                    subprocess.run(
                        [ffmpeg, "-y", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(wav), str(seg)],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        **_NO_CONSOLE
                    )

                except Exception:
                    log_exc(f"ffmpeg segment {i}")
                    continue
                try:
                    if (not seg.exists()) or (seg.stat().st_size < SAFE_MIN_AUDIO_BYTES):
                        continue
                except Exception:
                    continue

                txt = _safe_transcribe_seg(seg)
                if txt:
                    pieces.append(txt)

                pct = min(100, int(((start + length) / max(total, 1e-9)) * 100))
                ui_pct(f"Transcribing… {pct}%")
                _update_transcript_now()

            final_text = "\n".join(pieces).strip()
            if write_output:
                (TRANSCRIPTS_DIR / f"{src_path.stem}.txt").write_text(final_text, encoding="utf-8")
            if final_text:
                ui_transcript(final_text)
                ui_pct("Transcription complete.")
            else:
                elapsed = datetime.now().timestamp() - start_ts
                if elapsed >= NO_SPEECH_GRACE_SEC:
                    ui_pct("No speech")
                else:
                    ui_pct("Transcription complete.")
            ui_status("Transcription done")

    threading.Thread(target=run, daemon=True).start()

# ===================== GUI APP =====================
class DogVetApp:
    def __init__(self):
        self.SHUTTING_DOWN = False

        # runtime state
        self.files: List[Path] = []
        self.idx = 0
        self.current_media: Optional[Path] = None
        self.current_pos_ms = 0
        self.total_dur_ms = 0
        self.view_mode = "gallery"
        self.thumb_cache = {}
        self.gallery_items = []
        self.gallery_cells = {}
        self.gallery_resize_job = None
        self.last_thumb_w_used = None
        self.scrub_scale = None
        self.preview_container = None
        self.is_scrubbing = False

        # active gallery tab
        self.gallery_tab = "inbox"

        # VLC
        self.vlc_instance = None
        self.vlc_player = None

        # Tk
        self.root = Tk()
        self.root.title(f"{APP_NAME} {APP_VER} • VLC + Faster-Whisper")
        RIGHT_W = max(GALLERY_W_FALLBACK, PLAYER_W)
        self.root.geometry(f"{RIGHT_W + 460}x{PLAYER_H + 520}")
        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        Tk.report_callback_exception = self._tk_report_callback_exception

        # Vars
        self.status = StringVar(value="Ready")
        self.filename_var = StringVar(value="—")
        self.trans_pct = StringVar(value="")
        self.eta_var = StringVar(value="ETA —")
        self.trans_start_ts: Optional[float] = None

        # Named fonts (easy to resize these two lines later)
        self.font_trans_pct = Font(family="Segoe UI", size=10, slant="italic")  # progress text
        self.font_status    = Font(family="Segoe UI", size=9,  slant="italic")  # status line

        # Menubar
        self._build_menubar()

        # Top layout
        top = Frame(self.root)
        top.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)
        top.grid_columnconfigure(0, weight=0)
        top.grid_columnconfigure(1, weight=1)

        # Controls
        controls = Frame(top)
        controls.grid(row=0, column=0, sticky="nw", padx=(0, 12))
        controls.grid_columnconfigure(0, weight=0)
        controls.grid_columnconfigure(1, weight=1)  # allow some space for the bar

        Label(controls, text="Video file:", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky="w")
        Label(controls, textvariable=self.filename_var, wraplength=280, font=("Segoe UI", 10), fg="#333").grid(row=1, column=0, columnspan=3, sticky="w", pady=(0,6))

        # Percent text + progress bar + ETA on the same row
        Label(controls, textvariable=self.trans_pct, fg="#444", font=self.font_trans_pct).grid(
            row=2, column=0, sticky="w", padx=(0, 8)
        )

        # Progress bar styles for speed coloring
        self.pb_style = ttk.Style(self.root)
        self.pb_style.configure("SpeedFast.Horizontal.TProgressbar", background="#4CAF50")  # green
        self.pb_style.configure("SpeedMed.Horizontal.TProgressbar",  background="#FFC107")  # amber
        self.pb_style.configure("SpeedSlow.Horizontal.TProgressbar", background="#E53935")  # red

        self.progress = ttk.Progressbar(
            controls, mode="determinate", maximum=100, length=150,
            style="SpeedMed.Horizontal.TProgressbar"  # default mid speed
        )
        self.progress.grid(row=2, column=1, sticky="w")

        Label(controls, textvariable=self.eta_var, fg="#666", font=("Segoe UI", 9)).grid(
            row=2, column=2, sticky="w", padx=(8, 0)
        )

        Label(controls, textvariable=self.status, fg="#666", font=self.font_status).grid(row=3, column=0, columnspan=3, sticky="w", pady=(0,8))
        self._make_btn(controls, "Play", "#624185", self.play_action, 4, 0)
        self._make_btn(controls, "Stop", "#FFC107", self.stop_media, 4, 1)
        self._make_btn(controls, "Pause", "#FF7817", self.pause_media, 4, 2)
        self._make_btn(controls, "Skip",   "#2196F3", self.skip_item,    5, 0)
        self._make_btn(controls, "Allow",  "#4CAF50", self.approve,      5, 1)
        self._make_btn(controls, "Deny",   "#E53935", self.deny,         5, 2)

        controls.grid_columnconfigure(0, weight=1, uniform="buttons")
        controls.grid_columnconfigure(1, weight=1, uniform="buttons")
        controls.grid_columnconfigure(2, weight=1, uniform="buttons")

        controls.grid_rowconfigure(4, weight=1)
        controls.grid_rowconfigure(5, weight=1)

        # Right content holder
        self.right_top = Frame(top)
        self.right_top.grid(row=0, column=1, sticky="nsew")
        top.grid_rowconfigure(0, weight=1)
        self.right_top.grid_rowconfigure(0, weight=0)

        # Transcript
        bottom = Frame(self.root)
        bottom.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0,10))
        bottom.grid_columnconfigure(0, weight=1)
        bottom.grid_rowconfigure(1, weight=1)
        Label(bottom, text="TRANSCRIPT", font=("Segoe UI", 13, "bold")).grid(row=0, column=0, sticky="w", pady=(0,6))
        self.transcript_box = ScrolledText(bottom, height=12, wrap="word", font=TRANSCRIPT_FONT)
        self.transcript_box.tag_configure("bad", background="#FFCDD2", foreground="#B71C1C", underline=0)
        self.transcript_box.configure(state="disabled")
        self.transcript_box.grid(row=1, column=0, sticky="nsew")

        self.gallery_canvas = None
        self.gallery_inner = None
        self.gallery_inner_id = None
        self.gallery_vscroll = None

        # Start
        self.root.after(200, self.on_start)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI helpers ----------
    def _make_btn(self, parent, text, color, cmd, r, c):
        btn = Label(parent, text=text, bg=color, fg="white",
                    font=("Segoe UI", 13, "bold"),
                    width=11, height=3, bd=2, relief="ridge")  # uniform sizing
        btn.grid(row=r, column=c, padx=6, pady=6, sticky="nsew")
        btn.bind("<Button-1>", lambda e: cmd())


    def _build_menubar(self):
        menubar = Menu(self.root)

        folders_menu = Menu(menubar, tearoff=0)
        folders_menu.add_command(label="Set Inbox…", command=self.set_inbox_dir)
        folders_menu.add_command(label="Set Allow…", command=self.set_allow_dir)
        folders_menu.add_command(label="Set Deny…", command=self.set_deny_dir)
        folders_menu.add_separator()
        folders_menu.add_command(label="Reset to Defaults", command=self._reset_to_defaults)
        folders_menu.add_separator()
        folders_menu.add_command(label="Set/Change Password…", command=set_or_change_folders_password)
        menubar.add_cascade(label="Folders", menu=folders_menu)

        # Transcription menu for language selection
        trans_menu = Menu(menubar, tearoff=0)
        def _set_lang(lang_code: str):
            LANGUAGE["value"] = lang_code
            self.ui_set(self.status, f"Transcription language set to: {lang_code.upper() if lang_code!='auto' else 'AUTO'}")
            _save_config()
        trans_menu.add_command(label="Language: English",      command=lambda: _set_lang("en"))
        trans_menu.add_command(label="Language: Spanish",      command=lambda: _set_lang("es"))
        trans_menu.add_command(label="Language: Auto-detect",  command=lambda: _set_lang("auto"))
        menubar.add_cascade(label="Transcription", menu=trans_menu)

        help_menu = Menu(menubar, tearoff=0)
        help_menu.add_command(
            label="About",
            command=lambda: messagebox.showinfo(
                "About",
                f"{APP_NAME} {APP_VER}\nVLC A/V + Faster-Whisper transcription\nCreated by Angelo R. Dibello"
            )
        )
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    # ---------- Tk exception hook ----------
    def _tk_report_callback_exception(self, exc, val, tb):
        traceback.print_exception(exc, val, tb)
        try:
            if not self.SHUTTING_DOWN:
                messagebox.showerror("Error", f"{exc.__name__}: {val}")
        except Exception:
            pass

    # ---------- UI state helpers ----------
    def ui_call(self, fn, *args, **kwargs):
        try:
            if self.SHUTTING_DOWN: return
            self.root.after(0, lambda: (not self.SHUTTING_DOWN) and fn(*args, **kwargs))
        except Exception:
            pass

    def ui_set(self, var: StringVar, value: str):
        self.ui_call(var.set, value)

    def render_transcript(self, text: str):
        if self.SHUTTING_DOWN: return
        try:
            self.transcript_box.configure(state="normal")
            self.transcript_box.delete("1.0", "end")
            if text:
                if profanity:
                    for m in re.finditer(r"\w+|\W+", text):
                        tok = m.group(0)
                        norm = re.sub(r"^\W+|\W+$", "", tok).lower()
                        if norm and profanity.contains_profanity(norm):
                            self.transcript_box.insert("end", tok, ("bad",))
                        else:
                            self.transcript_box.insert("end", tok)
                else:
                    self.transcript_box.insert("end", text)
            self.transcript_box.configure(state="disabled")
        except Exception:
            pass

    def ui_update_transcript(self, text):
        self.ui_call(self.render_transcript, text)

    def clear_transcript(self):
        self.ui_update_transcript("")

    # Drive GUI progress bar + text + ETA (no console)
    def ui_set_progress_text(self, s: str):
        # update label
        self.ui_set(self.trans_pct, s)
        try:
            now = time()

            # Start timing when we see the first % update
            if self.trans_start_ts is None and re.search(r"\d+\s*%", s):
                self.trans_start_ts = now

            # Extract percent
            m = re.search(r"(\d+)\s*%", s)
            if m:
                pct = max(0, min(100, int(m.group(1))))
                self.ui_call(lambda: self.progress.config(value=pct))

                # ETA + speed color
                if self.trans_start_ts is not None and 0 < pct < 100:
                    elapsed = max(0.001, now - self.trans_start_ts)
                    rate_pct_per_s = pct / elapsed  # %/s
                    remaining_pct = 100 - pct
                    eta_s = remaining_pct / max(rate_pct_per_s, 0.001)

                    mm = int(eta_s // 60)
                    ss = int(round(eta_s % 60))
                    self.ui_set(self.eta_var, f"ETA {mm:02d}:{ss:02d}")

                    # thresholds: >1.2 fast; 0.5–1.2 medium; <0.5 slow
                    if rate_pct_per_s >= 1.2:
                        style = "SpeedFast.Horizontal.TProgressbar"
                    elif rate_pct_per_s >= 0.5:
                        style = "SpeedMed.Horizontal.TProgressbar"
                    else:
                        style = "SpeedSlow.Horizontal.TProgressbar"
                    self.ui_call(lambda st=style: self.progress.config(style=st))
                elif pct >= 100:
                    self.ui_set(self.eta_var, "ETA 00:00")
                    self.ui_call(lambda: self.progress.config(style="SpeedFast.Horizontal.TProgressbar"))
                else:
                    self.ui_set(self.eta_var, "ETA —")

            elif "complete" in s.lower():
                self.ui_call(lambda: self.progress.config(value=100))
                self.ui_set(self.eta_var, "ETA 00:00")
                self.ui_call(lambda: self.progress.config(style="SpeedFast.Horizontal.TProgressbar"))

            elif "no speech" in s.lower() or "error" in s.lower():
                self.ui_call(lambda: self.progress.config(value=0))
                self.ui_set(self.eta_var, "ETA —")
                self.ui_call(lambda: self.progress.config(style="SpeedSlow.Horizontal.TProgressbar"))

        except Exception:
            pass

    # ---------- VLC ----------
    def vlc_setup_once(self):
        if vlc is None:
            raise RuntimeError("python-vlc is not installed")
        if self.vlc_instance is None:
            self.vlc_instance = vlc.Instance("--no-video-title-show", "--quiet")
        if self.vlc_player is None:
            self.vlc_player = self.vlc_instance.media_player_new()

    def vlc_bind_to_widget(self, frame_widget):
        try:
            handle = frame_widget.winfo_id()
            if platform.system() == "Windows":
                self.vlc_player.set_hwnd(handle)
            elif platform.system() == "Darwin":
                self.vlc_player.set_nsobject(handle)
            else:
                self.vlc_player.set_xwindow(handle)
        except Exception:
            log_exc("vlc_bind_to_widget")

    def vlc_open_media(self, path: Path, start_ms: int = 0):
        self.vlc_setup_once()
        m = self.vlc_instance.media_new(str(path))
        self.vlc_player.set_media(m)
        self.total_dur_ms = 0
        try:
            m.parse_with_options(vlc.MediaParseFlag.local, timeout=1000)
            d = m.get_duration()
            if d and d > 0:
                self.total_dur_ms = d
        except Exception:
            pass
        if start_ms > 0:
            try:
                self.vlc_player.set_time(int(start_ms))
            except Exception:
                pass

    def vlc_play(self):
        self.vlc_setup_once()
        try:
            self.vlc_player.play()
        except Exception:
            log_exc("vlc_play")

    def vlc_pause(self):
        try:
            self.vlc_player.pause()
        except Exception:
            pass

    def vlc_stop(self):
        try:
            self.vlc_player.stop()
        except Exception:
            pass

    def vlc_get_time_ms(self) -> int:
        try:
            return max(0, int(self.vlc_player.get_time() or 0))
        except Exception:
            return 0

    def vlc_set_time_ms(self, ms: int):
        try:
            self.vlc_player.set_time(int(ms))
        except Exception:
            pass

    def vlc_get_length_ms(self) -> int:
        try:
            d = self.total_dur_ms or int(self.vlc_player.get_length() or 0)
            if d > 0:
                self.total_dur_ms = d
            return self.total_dur_ms
        except Exception:
            return self.total_dur_ms or 0

    # ---------- Gallery / Thumbs ----------
    def refresh_file_list(self) -> List[Path]:
        base = {
            "inbox": INBOX_DIR,
            "allow": ALLOW_DIR,
            "deny":  DENY_DIR,
        }.get(self.gallery_tab, INBOX_DIR)
        self.files = list_from_dir(base)
        return self.files

    def refresh_gallery(self):
        self.refresh_file_list()
        self.build_gallery_grid()
        self.ui_set(self.status, f"Refreshed {human_tab_name(self.gallery_tab)}")

    def compute_available_gallery_width(self) -> int:
        if not self.gallery_canvas:
            return GALLERY_W_FALLBACK
        canvas_w = self.gallery_canvas.winfo_width()
        if canvas_w <= 50:
            return GALLERY_W_FALLBACK
        sb_w = self.gallery_vscroll.winfo_width() if self.gallery_vscroll else 0
        return max(0, canvas_w - sb_w - 2)

    def calc_thumb_size(self, available_w: int) -> Tuple[int, int]:
        gaps = (VISIBLE_COLS - 1) * GALLERY_HPAD
        w = max(140, (available_w - gaps) // VISIBLE_COLS)
        h = int(round(w * 9 / 16))
        return w, h

    def make_thumb_image(self, path: Path, w: int, h: int):
        if not Image or not ImageTk:
            return None
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            return None
        key = (path, w, h)
        cached = self.thumb_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]

        img = Image.new("RGB", (w, h), "black")
        try:
            if path.suffix.lower() in VIDEO_EXTS and cv2 is not None:
                cap = cv2.VideoCapture(str(path))
                ok, frame = (cap.read() if cap.isOpened() else (False, None))
                cap.release()
                if ok and frame is not None and getattr(frame, "size", 0):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    fh, fw, _ = frame.shape
                    scale = min(w / fw, h / fh)
                    new_w, new_h = max(1, int(fw * scale)), max(1, int(fh * scale))
                    frame = cv2.resize(frame, (new_w, new_h))
                    img.paste(Image.fromarray(frame), ((w - new_w)//2, (h - new_h)//2))
            else:
                # audio placeholder
                for x in range(0, w, 6):
                    for y in range(h // 4, (3 * h) // 4, 4):
                        img.putpixel((min(x, w-1), min(y, h-1)), (80, 200, 120))
        except Exception:
            pass

        tk_img = ImageTk.PhotoImage(img)
        self.thumb_cache[key] = (mtime, tk_img)
        return tk_img

    def update_gallery_highlight(self):
        for i, cell in self.gallery_cells.items():
            if i == self.idx:
                cell.configure(highlightthickness=3, highlightbackground="#FF9800", bd=1, relief="solid")
            else:
                cell.configure(highlightthickness=1, highlightbackground="#CCCCCC", bd=1, relief="solid")

    def build_gallery_grid(self):
        if self.SHUTTING_DOWN: return
        for wdg in self.right_top.winfo_children():
            wdg.destroy()
        self.gallery_items.clear()
        self.gallery_cells.clear()

        header = Frame(self.right_top)
        header.pack(fill="x")

        Label(header, text=f"VIDEOS (newest first) — {self.gallery_tab.upper()}",
              font=("Segoe UI", 11, "bold")).pack(side="left")

        actions = Frame(header)
        actions.pack(side="right")
        Button(actions, text="Refresh", command=self.refresh_gallery).pack(side="right", padx=(8, 0))

        counts = get_tab_counts()
        tabs = Frame(actions)
        tabs.pack(side="right")

        def _switch_tab(tabkey: str):
            if self.gallery_tab == tabkey:
                return
            self.gallery_tab = tabkey
            self.refresh_file_list()
            self.build_gallery_grid()

        def _make_tab(label: str, key: str, count: int):
            is_active = (self.gallery_tab == key)
            bg = "#1976D2" if is_active else "#EEEEEE"
            fg = "white" if is_active else "#333333"
            bd = 2 if is_active else 1
            relief = "ridge" if is_active else "solid"
            text = f"{label} ({count})"
            t = Label(tabs, text=text, padx=10, pady=4, bg=bg, fg=fg, bd=bd, relief=relief, font=("Segoe UI", 10, "bold"))
            t.pack(side="left", padx=4)
            t.bind("<Button-1>", lambda e, k=key: _switch_tab(k))
            try:
                t.configure(cursor="hand2")
            except Exception:
                pass

        _make_tab("Inbox", "inbox", counts["inbox"])
        _make_tab("Allow", "allow", counts["allow"])
        _make_tab("Deny",  "deny",  counts["deny"])

        self.gallery_canvas = Canvas(self.right_top, highlightthickness=0)
        self.gallery_vscroll = Scrollbar(self.right_top, orient="vertical", command=self.gallery_canvas.yview)
        self.gallery_canvas.configure(yscrollcommand=self.gallery_vscroll.set)
        self.gallery_canvas.pack(side="left", fill="both", expand=True, pady=(6,0))
        self.gallery_vscroll.pack(side="right", fill="y", pady=(6,0))

        self.gallery_inner = Frame(self.gallery_canvas)
        self.gallery_inner_id = self.gallery_canvas.create_window((0, 0), window=self.gallery_inner, anchor="nw")

        def _on_inner_config(_e):
            if self.SHUTTING_DOWN: return
            self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
            c_w = self.gallery_canvas.winfo_width()
            if c_w > 0:
                self.gallery_canvas.itemconfigure(self.gallery_inner_id, width=c_w)
        self.gallery_inner.bind("<Configure>", _on_inner_config)

        # mouse wheel
        try:
            self.root.unbind_all("<MouseWheel>")
            self.root.unbind_all("<Button-4>")
            self.root.unbind_all("<Button-5>")
        except Exception:
            pass
        def _on_mousewheel(event):
            if self.SHUTTING_DOWN: return
            delta = -1 if (event.delta or 0) > 0 else 1
            self.gallery_canvas.yview_scroll(delta, "units")
        self.root.bind_all("<MouseWheel>", _on_mousewheel)
        self.root.bind_all("<Button-4>", lambda e: self.gallery_canvas.yview_scroll(-1, "units"))
        self.root.bind_all("<Button-5>", lambda e: self.gallery_canvas.yview_scroll(1, "units"))

        self.right_top.update_idletasks()
        cw = self.gallery_canvas.winfo_width()
        if cw < 100:
            self.root.after(40, self.build_gallery_grid)
            return

        self.gallery_canvas.bind("<Configure>", self.on_gallery_configure)
        self.right_top.bind("<Configure>", self.on_gallery_configure)

        avail_w = self.compute_available_gallery_width()
        thumb_w, thumb_h = self.calc_thumb_size(avail_w)
        visible_height = (thumb_h + THUMB_LABEL_H + 2*GALLERY_VPAD) * VISIBLE_ROWS + 4
        self.gallery_canvas.config(height=visible_height)

        # Empty state
        if not self.files:
            for c in range(VISIBLE_COLS):
                self.gallery_inner.grid_columnconfigure(c, weight=1)
            empty = Frame(self.gallery_inner)
            empty.grid(row=0, column=0, columnspan=VISIBLE_COLS, pady=20, sticky="nsew")
            Label(empty, text=f"{human_tab_name(self.gallery_tab)} Empty", font=("Segoe UI", 13, "bold"), fg="#666").pack(pady=(8,4))
            Label(empty, text="Drop videos in the selected folder to see them here.", font=("Segoe UI", 10), fg="#888").pack()
            self.idx = 0
            self.current_media = None
            self.right_top.update_idletasks()
            self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
            return

        # Fixed grid with spacers
        real_count = len(self.files)
        target_count = max(real_count, VISIBLE_ROWS * VISIBLE_COLS)
        for i in range(target_count):
            r, c = divmod(i, VISIBLE_COLS)
            if i >= real_count:
                cell = Frame(self.gallery_inner, bd=0, relief="flat", highlightthickness=0)
                cell.grid(row=r, column=c, padx=GALLERY_HPAD, pady=GALLERY_VPAD, sticky="nsew")
                spacer = Frame(cell, width=thumb_w, height=thumb_h + THUMB_LABEL_H + 26, bg=self.root.cget("bg"))
                spacer.pack()
                continue

            p = self.files[i]
            if p.suffix.lower() not in (VIDEO_EXTS | AUDIO_EXTS):
                continue

            thumb = self.make_thumb_image(p, thumb_w, thumb_h)
            if not thumb and Image and ImageTk:
                img = Image.new("RGB", (thumb_w, thumb_h), "black")
                thumb = ImageTk.PhotoImage(img)

            cell = Frame(self.gallery_inner, bd=1, relief="solid", highlightthickness=1, highlightbackground="#CCCCCC")
            cell.grid(row=r, column=c, padx=GALLERY_HPAD, pady=GALLERY_VPAD, sticky="nsew")

            if thumb:
                img_lbl = Label(cell, image=thumb, bg="black", width=thumb_w, height=thumb_h)
                img_lbl.pack()
                self.gallery_items.append(thumb)
            else:
                Label(cell, text="(thumbnail)", width=thumb_w, height=int(thumb_h/20), bg="black", fg="white").pack()

            name = Label(cell, text=p.name, font=("Segoe UI", 10), wraplength=thumb_w, anchor="center", justify="center")
            name.pack(pady=(2,2))
            try:
                mtime = p.stat().st_mtime
                dt_str = datetime.fromtimestamp(mtime).strftime("%b %d, %Y • %H:%M")
            except Exception:
                dt_str = ""
            Label(cell, text=dt_str, font=("Segoe UI", 9, "italic"), fg="#666",
                  wraplength=thumb_w, anchor="center", justify="center").pack(pady=(0, 4))
            self.gallery_cells[i] = cell

            def _select(i=i, open_player=True):
                self.idx = i
                self.update_gallery_highlight()
                if open_player:
                    self.show_player(i)

            def _hover_in(ev=None, i=i):
                self.idx = i
                self.update_gallery_highlight()

            cell.bind("<Enter>", _hover_in)
            cell.bind("<Button-1>", lambda e, i=i: _select(i, True))
            name.bind("<Button-1>", lambda e, i=i: _select(i, True))
            if thumb:
                img_lbl.bind("<Button-1>", lambda e, i=i: _select(i, True))
                try: img_lbl.configure(cursor="hand2")
                except Exception: pass
            try:
                cell.configure(cursor="hand2")
                name.configure(cursor="hand2")
            except Exception:
                pass

        for c in range(VISIBLE_COLS):
            self.gallery_inner.grid_columnconfigure(c, weight=1)

        self.right_top.update_idletasks()
        self.gallery_canvas.configure(scrollregion=self.gallery_canvas.bbox("all"))
        self.update_gallery_highlight()

    def on_gallery_configure(self, event=None):
        if self.SHUTTING_DOWN or self.view_mode != "gallery" or self.gallery_canvas is None:
            return
        avail_w = self.compute_available_gallery_width()
        thumb_w, _ = self.calc_thumb_size(avail_w)
        if self.last_thumb_w_used == thumb_w:
            try:
                self.gallery_canvas.itemconfigure(self.gallery_inner_id, width=self.gallery_canvas.winfo_width())
            except Exception:
                pass
            return
        self.last_thumb_w_used = thumb_w
        def _rebuild():
            if self.SHUTTING_DOWN: return
            if self.view_mode == "gallery":
                self.build_gallery_grid()
        if self.gallery_resize_job is not None:
            try: self.root.after_cancel(self.gallery_resize_job)
            except Exception: pass
        self.gallery_resize_job = self.root.after(120, _rebuild)

    # ---------- Flow ----------
    def load_item(self, i):
        if not self.files:
            self.current_media = None
            self.ui_set(self.status, "No files found")
            self.clear_transcript()
            return
        self.idx = i % len(self.files)
        self.current_media = self.files[self.idx]
        self.ui_set(self.filename_var, self.current_media.name)
        self.current_pos_ms = 0
        self.clear_transcript()
        self.ui_set(self.trans_pct, "")

        if self.view_mode == "player":
            if self.preview_container is not None:
                self.vlc_open_media(self.current_media, start_ms=0)
                self.vlc_bind_to_widget(self.preview_container)
            self.play_media()

            # reset ETA/progress timing
            self.trans_start_ts = None
            self.eta_var.set("ETA —")
            self.ui_set_progress_text("Transcribing… 0%")

            transcribe_file(
                self.current_media,
                ui_status=lambda s: self.ui_set(self.status, s),
                ui_pct=self.ui_set_progress_text,        # drive GUI progress bar + ETA
                ui_transcript=self.ui_update_transcript
            )
        else:
            self.update_gallery_highlight()

    def next_item(self):
        self.load_item(self.idx + 1)

    # ---------- Decisions ----------
    def approve(self):
        if not self.current_media: return
        if self.current_media.parent == ALLOW_DIR:
            self.ui_set(self.status, "Already in allow/")
            if self.files: self.next_item()
            else: self.show_gallery()
            return
        dest = ALLOW_DIR / self.current_media.name
        self.pause_media(); self.stop_media()
        try:
            shutil.move(str(self.current_media), str(dest))
        except Exception as e:
            self.ui_call(messagebox.showerror, "Move Error", str(e)); return
        append_decision(self.current_media.name, "allow")
        self.ui_set(self.status, "Approved → moved to allow/")
        self.refresh_file_list()
        self.clear_transcript()
        if self.files: self.next_item()
        else: self.show_gallery()

    def deny(self):
        if not self.current_media: return
        if self.current_media.parent == DENY_DIR:
            self.ui_set(self.status, "Already in deny/")
            if self.files: self.next_item()
            else: self.show_gallery()
            return
        dest = DENY_DIR / self.current_media.name
        self.pause_media(); self.stop_media()
        try:
            shutil.move(str(self.current_media), str(dest))
        except Exception as e:
            self.ui_call(messagebox.showerror, "Move Error", str(e)); return
        append_decision(self.current_media.name, "deny")
        self.ui_set(self.status, "Denied → moved to deny/")
        self.refresh_file_list()
        self.clear_transcript()
        if self.files: self.next_item()
        else: self.show_gallery()

    def skip_item(self):
        if not self.files: return
        if self.view_mode == "player":
            self.next_item()
        else:
            self.show_player((self.idx + 1) % len(self.files))

    # ---------- View switching ----------
    def show_gallery(self):
        self.view_mode = "gallery"
        self.pause_media(); self.stop_media()
        self.scrub_scale = None
        self.preview_container = None
        self.last_thumb_w_used = None
        self.refresh_file_list()
        if not self.files:
            self.idx = 0
            self.current_media = None
        self.root.after_idle(self.build_gallery_grid)

    def show_player(self, index: int):
        if not self.files:
            self.ui_set(self.status, f"No videos to play in {human_tab_name(self.gallery_tab)}")
            self.view_mode = "gallery"
            return

        self.view_mode = "player"
        try:
            self.right_top.unbind("<Configure>")
            if self.gallery_canvas is not None:
                self.gallery_canvas.unbind("<Configure>")
            self.root.unbind_all("<MouseWheel>")
            self.root.unbind_all("<Button-4>")
            self.root.unbind_all("<Button-5>")
        except Exception:
            pass

        for wdg in self.right_top.winfo_children():
            wdg.destroy()

        header = Frame(self.right_top)
        header.pack(fill="x", pady=(0,6))
        Button(header, text="⟵ Back to Gallery", command=self.show_gallery).pack(side="left")
        Label(header, textvariable=self.filename_var, font=("Segoe UI", 11, "bold")).pack(side="left", padx=10)

        player_area = Frame(self.right_top)
        player_area.pack()

        preview_container_local = Frame(player_area, width=PLAYER_W, height=PLAYER_H, bg="black")
        preview_container_local.pack()
        preview_container_local.update_idletasks()
        self.preview_container = preview_container_local

        s = Scale(player_area, from_=0.0, to=1.0, orient=HORIZONTAL, length=PLAYER_W, showvalue=False, resolution=0.01)
        s.pack(pady=(6, 0))
        Label(player_area, text="Move the slider to scrub through", font=("Segoe UI", 12, "italic"), fg="#666").pack(pady=(2, 6))
        s.bind("<Button-1>", self.on_scrub_press)
        s.bind("<ButtonRelease-1>", self.on_scrub_release)
        self.scrub_scale = s

        index = min(max(0, index), len(self.files) - 1)
        self.load_item(index)

    # ---------- Playback ----------
    def play_action(self):
        if not self.files:
            self.ui_set(self.status, f"No videos in {human_tab_name(self.gallery_tab)}")
            return
        if self.view_mode == "gallery":
            safe_idx = min(self.idx, len(self.files) - 1)
            self.show_player(safe_idx)
            return
        self.play_media()

    def play_media(self):
        if not self.current_media or self.SHUTTING_DOWN: return
        self.vlc_setup_once()
        state = self.vlc_player.get_state()
        if state in (vlc.State.NothingSpecial, vlc.State.Stopped, vlc.State.Ended):
            self.vlc_open_media(self.current_media, start_ms=self.current_pos_ms)
            if self.preview_container is not None:
                self.vlc_bind_to_widget(self.preview_container)
        self.vlc_play()
        self.ui_set(self.status, f"Playing: {self.current_media.name}")
        self.root.after(150, self.poll_clock_update)

    def pause_media(self):
        self.vlc_pause()
        self.ui_set(self.status, "Paused")

    def stop_media(self):
        self.vlc_stop()
        self.current_pos_ms = 0
        self.ui_set(self.status, "Stopped")

    def poll_clock_update(self):
        if self.SHUTTING_DOWN or self.view_mode != "player" or self.scrub_scale is None:
            return
        try:
            length = self.vlc_get_length_ms()
            if length and length > 0:
                if float(self.scrub_scale.cget("to")) != float(length/1000.0):
                    self.scrub_scale.configure(to=length/1000.0)
            if not self.is_scrubbing:
                t_ms = self.vlc_get_time_ms()
                self.scrub_scale.set(t_ms / 1000.0)
        except Exception:
            pass
        if self.view_mode == "player":
            self.root.after(150, self.poll_clock_update)

    # ---------- Scrub ----------
    def on_scrub_press(self, _):
        self.is_scrubbing = True

    def on_scrub_release(self, _):
        self.is_scrubbing = False
        if not self.current_media or self.SHUTTING_DOWN or self.scrub_scale is None:
            return
        target_s = float(self.scrub_scale.get())
        self.current_pos_ms = max(0, int(target_s * 1000.0))
        self.vlc_set_time_ms(self.current_pos_ms)

    # ---------- Folder ops ----------
    def set_inbox_dir(self):
        global INBOX_DIR
        if not require_folders_password():
            return
        d = askdirectory(initialdir=str(INBOX_DIR), title="Select Inbox (media) folder")
        if not d:
            return
        INBOX_DIR = _ensure_dir(Path(d))
        self.refresh_gallery()
        self.ui_set(self.status, f"Inbox set to: {INBOX_DIR}")
        _save_config()

    def set_allow_dir(self):
        global ALLOW_DIR
        if not require_folders_password():
            return
        d = askdirectory(initialdir=str(ALLOW_DIR), title="Select ALLOW folder")
        if not d:
            return
        ALLOW_DIR = _ensure_dir(Path(d))
        if self.gallery_tab == "allow":
            self.refresh_gallery()
        self.ui_set(self.status, f"Allow folder set to: {ALLOW_DIR}")
        _save_config()

    def set_deny_dir(self):
        global DENY_DIR
        if not require_folders_password():
            return
        d = askdirectory(initialdir=str(DENY_DIR), title="Select DENY folder")
        if not d:
            return
        DENY_DIR = _ensure_dir(Path(d))
        if self.gallery_tab == "deny":
            self.refresh_gallery()
        self.ui_set(self.status, f"Deny folder set to: {DENY_DIR}")
        _save_config()

    def _reset_to_defaults(self):
        global INBOX_DIR, ALLOW_DIR, DENY_DIR
        if not require_folders_password():
            return
        INBOX_DIR = _ensure_dir(APP_ROOT / "All_media")
        ALLOW_DIR = _ensure_dir(APP_ROOT / "allow")
        DENY_DIR = _ensure_dir(APP_ROOT / "deny")
        _save_config()
        self.refresh_gallery()
        self.ui_set(self.status, "Folders reset to APP_ROOT defaults")

    # ---------- Lifecycle ----------
    def on_start(self):
        if profanity:
            try:
                profanity.load_censor_words()
                if BADWORDS_TXT.exists():
                    extra = [ln.strip().lower() for ln in BADWORDS_TXT.read_text(encoding="utf-8").splitlines() if ln.strip()]
                    if extra:
                        profanity.load_censor_words(extra_words=extra)
            except Exception:
                pass

        _load_config()
        self.refresh_file_list()
        self.show_gallery()
        if not self.files:
            self.ui_set(self.status, "No media files found")
        self.clear_transcript()

    def on_close(self):
        if self.SHUTTING_DOWN: return
        self.SHUTTING_DOWN = True
        try:
            try:
                self.right_top.unbind("<Configure>")
                if self.gallery_canvas is not None:
                    self.gallery_canvas.unbind("<Configure>")
                self.root.unbind_all("<MouseWheel>")
                self.root.unbind_all("<Button-4>")
                self.root.unbind_all("<Button-5>")
            except Exception:
                pass
            try: self.pause_media()
            except Exception: pass
            try: self.stop_media()
            except Exception: pass
            try: self.gallery_items.clear(); self.thumb_cache.clear()
            except Exception: pass
        finally:
            try: self.root.destroy()
            except Exception: pass
            os._exit(0)

def main():
    app = DogVetApp()
    app.root.mainloop()

if __name__ == "__main__":
    main()
