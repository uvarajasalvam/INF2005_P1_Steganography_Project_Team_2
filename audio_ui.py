# audio_ui.py  —  WAV LSB stego + optional MP3Stego backend + MP3 method toggle
import os, math, zlib, hmac, hashlib, struct, base64, wave as wave_mod, subprocess, shutil, tempfile, uuid
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.random import PCG64, Generator

from PySide6 import QtCore
from PySide6.QtCore import Qt, Signal
import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtWidgets
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QFileDialog, QLineEdit, QFormLayout, QTabWidget, QSplitter,
    QMessageBox, QTextEdit, QSlider, QSpinBox, QDoubleSpinBox, QCheckBox
)

from pydub import AudioSegment
from pydub.utils import which as which_ffmpeg
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import soundfile as sf
import sys, re

print("FFmpeg =", which_ffmpeg("ffmpeg"))

# ----------------- constants / utils -----------------
SAFE_OUTDIR = Path.cwd() / "output"
SAFE_OUTDIR.mkdir(exist_ok=True)

def _safe_filename(name: str, fallback: str = "payload.bin") -> str:
    cand = (name or fallback).replace("\\", "_").replace("/", "_").strip()
    return cand or fallback

class PCM(NamedTuple):
    samples_i16: np.ndarray
    rate: int
    channels: int
    sampwidth: int
    src_path: str
    src_ext: str
    is_lossy: bool

LOSSY_EXTS    = {".mp3", ".aac", ".m4a", ".ogg", ".opus", ".wma"}
LOSSLESS_EXTS = {".wav", ".aif", ".aiff", ".flac", ".alac"}

# ----------------- decoding any to PCM (for view) -----------------
def decode_any_to_pcm(path: str, force_mono: bool = True) -> PCM:
    ext = Path(path).suffix.lower()
    try:
        a = AudioSegment.from_file(path)
        if force_mono:
            a = a.set_channels(1)
        rate = a.frame_rate
        ch = a.channels
        arr = np.array(a.get_array_of_samples(), dtype=np.int16)
        if ch > 1:
            arr = arr.reshape(-1, ch)
        return PCM(arr, rate, ch, 2, path, ext, ext in LOSSY_EXTS)
    except Exception:
        data, rate = sf.read(path, dtype="int16", always_2d=True)
        if force_mono and data.shape[1] > 1:
            data = data.mean(axis=1).astype(np.int16, copy=False).reshape(-1, 1)
        ch = data.shape[1]
        arr = data[:,0] if force_mono else data
        return PCM(arr, rate, ch, 2, path, ext, ext in LOSSY_EXTS)

def write_pcm_as_wav(out_path: str, pcm: PCM):
    samples = pcm.samples_i16
    rate = pcm.rate
    if samples.ndim == 1:
        sf.write(out_path, samples, rate, format="WAV", subtype="PCM_16")
    else:
        sf.write(out_path, samples.astype(np.int16, copy=False), rate, format="WAV", subtype="PCM_16")

# ---------- File types ----------
AUDIO_EXTS = {".wav"}
SUPPORTED_AUDIO_EXTS = AUDIO_EXTS.union({".mp3", ".flac", ".ogg"})

# ---------- small helpers ----------
def human_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    for unit in ["KB","MB","GB","TB"]:
        n /= 1024.0
        if n < 1024: return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"

def fmt_time(samples: int, rate: int) -> str:
    if rate <= 0: return "0:00"
    secs = max(0, samples) / rate
    m = int(secs // 60); s = int(round(secs % 60))
    if s == 60: m += 1; s = 0
    return f"{m}:{s:02d}"

def secs_to_frames(start_sec: float, dur_sec: float, rate: int, total_frames: int) -> tuple[int, int]:
    if rate <= 0 or total_frames <= 0: return 0, 1
    start = int(math.floor(max(0.0, start_sec) * rate))
    end   = int(math.ceil(max(start_sec + max(dur_sec, 0.0), 1e-9) * rate))
    start = min(start, max(0, total_frames - 1))
    if end <= start: end = start + 1
    end = min(end, total_frames)
    return start, end - start

def sha256(b: bytes) -> bytes: return hashlib.sha256(b).digest()

def hkdf_sha256(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    prk = hmac.new(salt, key, hashlib.sha256).digest()
    okm = b""; t = b""; i = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t; i += 1
    return okm[:length]

def cover_fingerprint(path: str, n_bytes: int = 1_048_576) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = n_bytes
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk: break
            h.update(chunk); remaining -= len(chunk)
    return h.digest()[:16]

def canonical_salt(lsb: int, roi_xywh: tuple[int,int,int,int], cover_id: bytes, media_kind: str) -> bytes:
    x0, y0, w, h = roi_xywh
    s = f"kind:{media_kind}|roi:{x0},{y0},{w},{h}|lsb:{lsb}|cover_id".encode()
    return sha256(s + cover_id)

def kdf_from_key(user_key: str, salt: bytes) -> dict:
    key_bytes = user_key.encode("utf-8")
    okm = hkdf_sha256(key_bytes, salt, b"stego-hkdf-v1", 16 + 1 + 32 + 4 + 12)
    off = 0
    K_perm = okm[off:off+16]; off += 16
    K_bit  = okm[off:off+1];  off += 1
    K_crypto = okm[off:off+32]; off += 32
    K_check  = okm[off:off+4]; off += 4
    nonce    = okm[off:off+12]; off += 12
    return {"K_perm": K_perm, "K_bit": K_bit, "K_crypto": K_crypto, "K_check": K_check, "nonce": nonce}

# ---------------- header pack/unpack ----------------
HEADER_WO_CRC_LEN = 80
HEADER_TOTAL_LEN  = 84
HEADER_FMT_PREFIX = b"STG1"

def build_header(version: int, lsb: int, roi_xywh: tuple[int,int,int,int],
                 payload_len: int, cover_fp16: bytes, salt16: bytes,
                 nonce12: bytes, kcheck4: bytes) -> bytes:
    magic = HEADER_FMT_PREFIX; flags = 0; pad = 0
    x0, y0, w, h = roi_xywh
    header_wo_crc = (
        magic +
        struct.pack("<BBBB", version, flags, lsb, pad) +
        struct.pack("<IIII", x0, y0, w, h) +
        struct.pack("<Q", payload_len) +
        cover_fp16 + salt16 + nonce12 + kcheck4
    )
    crc = zlib.crc32(header_wo_crc) & 0xFFFFFFFF
    return header_wo_crc + struct.pack("<I", crc)

class ParsedHeader:
    __slots__ = ("version","flags","lsb","roi","payload_len","cover_fp16","salt16","nonce12","kcheck4","crc32")
    def __init__(self, version, flags, lsb, roi, payload_len, cover_fp16, salt16, nonce12, kcheck4, crc32):
        self.version=version; self.flags=flags; self.lsb=lsb; self.roi=roi; self.payload_len=payload_len
        self.cover_fp16=cover_fp16; self.salt16=salt16; self.nonce12=nonce12; self.kcheck4=kcheck4; self.crc32=crc32

def parse_header(buf: bytes) -> ParsedHeader:
    if len(buf) != HEADER_TOTAL_LEN: raise ValueError("Header size mismatch")
    if buf[:4] != HEADER_FMT_PREFIX: raise ValueError("Bad header magic")
    version, flags, lsb, pad = struct.unpack("<BBBB", buf[4:8])
    x0, y0, w, h = struct.unpack("<IIII", buf[8:24])
    payload_len, = struct.unpack("<Q", buf[24:32])
    cover_fp16 = buf[32:48]; salt16 = buf[48:64]; nonce12 = buf[64:76]; kcheck4 = buf[76:80]
    crc32_read, = struct.unpack("<I", buf[80:84])
    calc = zlib.crc32(buf[:80]) & 0xFFFFFFFF
    if calc != crc32_read: raise ValueError("Header CRC32 mismatch")
    return ParsedHeader(version, flags, lsb, (x0,y0,w,h), payload_len, cover_fp16, salt16, nonce12, kcheck4, crc32_read)

# -------- payload meta (TLV) --------
def build_meta(mode: int, filename: str | None, mime: str | None) -> bytes:
    parts = []
    if filename:
        b = filename.encode("utf-8"); parts.append(b"\x01" + struct.pack("<H", len(b)) + b)
    if mime:
        b = mime.encode("utf-8");   parts.append(b"\x02" + struct.pack("<H", len(b)) + b)
    parts.append(b"\x03" + struct.pack("<H", 1) + bytes([mode & 0xFF]))
    return b"".join(parts)

def parse_meta(buf: bytes) -> dict:
    i = 0; out = {"filename": None, "mime": None, "mode": 0}
    while i + 3 <= len(buf):
        t = buf[i]; n = struct.unpack("<H", buf[i+1:i+3])[0]; i += 3
        v = buf[i:i+n]; i += n
        if   t == 0x01: out["filename"] = v.decode("utf-8", errors="ignore")
        elif t == 0x02: out["mime"]     = v.decode("utf-8", errors="ignore")
        elif t == 0x03 and n >= 1: out["mode"] = int(v[0])
    return out

# -------- key token helpers --------
def make_key_token(media_kind: str, lsb: int, roi_xywh: tuple[int,int,int,int], salt16: bytes, kcheck4: bytes) -> str:
    media_code = 0 if media_kind == "image" else 1
    x0, y0, w, h = roi_xywh
    packed = struct.pack("<4sBBIIII16s4s", b"KEY1", media_code, lsb, x0, y0, w, h, salt16, kcheck4)
    b64 = base64.urlsafe_b64encode(packed).decode("ascii")
    return f"stg1:{b64}"

def parse_key_token(token: str) -> dict:
    if not token.startswith("stg1:"): raise ValueError("Invalid token prefix")
    raw = base64.urlsafe_b64decode(token.split("stg1:",1)[1].encode("ascii"))
    magic, media_code, lsb, x0, y0, w, h, salt16, kcheck4 = struct.unpack("<4sBBIIII16s4s", raw)
    if magic != b"KEY1": raise ValueError("Bad token magic")
    media_kind = "image" if media_code == 0 else "audio"
    return {"media_kind": media_kind, "lsb": lsb, "roi": (x0,y0,w,h), "salt16": salt16, "kcheck4": kcheck4}

# ----------------- waveform widget -----------------
class WaveformView(QWidget):
    selectionChanged = Signal(int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._rate = 1; self._x = None; self._y = None; self._span = None; self._press_sample = None
        self._impact_patches = []
        lay = QVBoxLayout(self)
        self.fig = Figure(figsize=(10, 4), facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig); lay.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111); self._pretty_axes(); self.fig.tight_layout()
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _pretty_axes(self):
        self.ax.set_facecolor("#f9fbff"); self.ax.grid(True, alpha=0.25)
        for s in ["top","right"]: self.ax.spines[s].set_visible(False)
        self.ax.set_xlabel("Time (s)"); self.ax.set_ylabel("Amplitude")

    def set_audio(self, samples_mono: np.ndarray, rate: int):
        self._rate = int(rate); n = len(samples_mono)
        if n == 0: self.clear(); return
        step = max(1, n // 200_000)
        y = samples_mono[::step].astype(np.float32)
        maxv = float(np.max(np.abs(y))) if y.size else 1.0
        if maxv <= 0: maxv = 1.0
        y = y / maxv
        x = (np.arange(len(y)) * step) / self._rate
        self._x, self._y = x, y
        self.ax.clear(); self._clear_impacts(); self._pretty_axes()
        self.ax.plot(self._x, self._y, linewidth=1.25)
        self.ax.set_xlim(float(self._x[0]), float(self._x[-1])); self.ax.set_ylim(-1.1, 1.1)
        self._remove_span(); self.canvas.draw_idle()

    def set_selection(self, start_sample: int, length: int):
        if self._rate <= 0 or self._x is None: return
        start_sample = max(0, start_sample)
        end_sample = max(start_sample + 1, start_sample + length)
        t0 = start_sample / self._rate; t1 = end_sample / self._rate
        self._draw_span(t0, t1)

    def _on_press(self, ev):
        if ev.inaxes != self.ax or self._rate <= 0: return
        self._press_sample = int(max(0.0, ev.xdata) * self._rate)

    def _on_move(self, ev):
        if self._press_sample is None or ev.inaxes != self.ax: return
        cur_sample = int(max(0.0, ev.xdata) * self._rate)
        s0 = min(self._press_sample, cur_sample); s1 = max(self._press_sample, cur_sample)
        self._draw_span(s0 / self._rate, s1 / self._rate)

    def _on_release(self, ev):
        if self._press_sample is None or ev.inaxes != self.ax:
            self._press_sample = None; return
        cur_sample = int(max(0.0, ev.xdata) * self._rate)
        s0 = max(0, min(self._press_sample, cur_sample)); s1 = max(s0 + 1, max(self._press_sample, cur_sample))
        self._press_sample = None; self.selectionChanged.emit(s0, s1 - s0)

    def _draw_span(self, t0: float, t1: float):
        self._remove_span()
        if t1 <= t0: return
        self._span = self.ax.axvspan(t0, t1, alpha=0.25)
        self.canvas.draw_idle()

    def _remove_span(self):
        if self._span is not None:
            try: self._span.remove()
            except Exception: pass
            self._span = None

    def _clear_impacts(self):
        for p in self._impact_patches:
            try: p.remove()
            except Exception: pass
        self._impact_patches = []; self.canvas.draw_idle()

    def highlight_impacts(self, frames: np.ndarray, rate: int):
        if rate <= 0 or frames is None or frames.size == 0:
            self._clear_impacts(); return
        f = np.unique(frames.astype(np.int64)); runs = []; start = f[0]; prev = f[0]
        for x in f[1:]:
            if x == prev + 1: prev = x
            else: runs.append((start, prev)); start = prev = x
        runs.append((start, prev))
        MAX_RUNS = 1200
        if len(runs) > MAX_RUNS:
            step = int(np.ceil(len(runs) / MAX_RUNS)); runs = runs[::step]
        self._clear_impacts()
        for a, b in runs:
            t0 = a / rate; t1 = (b + 1) / rate
            patch = self.ax.axvspan(t0, t1, alpha=0.25, zorder=0.5)
            self._impact_patches.append(patch)
        self.canvas.draw_idle()

    def clear(self):
        self.ax.clear(); self._pretty_axes(); self._clear_impacts(); self.canvas.draw_idle()
        self._x = self._y = None; self._remove_span()

    def set_comparison(self, samples_orig: np.ndarray, samples_stego: np.ndarray):
        from matplotlib.gridspec import GridSpec
        self.fig.clf()
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
        n = min(len(samples_orig), len(samples_stego))
        max_o = float(np.max(np.abs(samples_orig[:n]))) if n else 1.0
        max_s = float(np.max(np.abs(samples_stego[:n]))) if n else 1.0
        if max_o == 0: max_o = 1.0
        if max_s == 0: max_s = 1.0
        orig_norm  = samples_orig[:n]  / max_o
        stego_norm = samples_stego[:n] / max_s
        t = np.arange(n) / max(1, self._rate)
        ax_o = self.fig.add_subplot(gs[0]); ax_s = self.fig.add_subplot(gs[1], sharex=ax_o)
        ax_o.plot(t, orig_norm, linewidth=1); ax_o.set_ylabel("Original"); ax_o.grid(True, alpha=0.25)
        ax_s.plot(t, stego_norm, linewidth=1); ax_s.set_ylabel("Stego"); ax_s.set_xlabel("Time (s)"); ax_s.grid(True, alpha=0.25)
        for ax in (ax_o, ax_s):
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False); ax.set_xlim(t[0], t[-1]); ax.set_ylim(-1.1, 1.1)
        self.ax = ax_s; self.canvas.draw_idle()

# ----------------- drag label -----------------
class DropLabel(QLabel):
    fileDropped = Signal(str)
    def __init__(self, title: str, exts: set[str] | None, parent=None):
        super().__init__(parent)
        self.exts = exts; self.setAcceptDrops(True); self.setAlignment(Qt.AlignCenter)
        self.setText(f"Drop {title} here\n—or—\nClick to browse"); self.setObjectName("DropLabel")
        self.setStyleSheet("""
            #DropLabel { border: 2px dashed #888; border-radius: 12px; padding: 16px;
                         color: #444; font-size: 14px; background: #fafafa; }
            #DropLabel:hover { background: #f0f7ff; }
        """)

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            path = e.mimeData().urls()[0].toLocalFile()
            if self._acceptable(path): e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        path = e.mimeData().urls()[0].toLocalFile()
        if self._acceptable(path): self.fileDropped.emit(path)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == Qt.LeftButton:
            dlg = QFileDialog(self); dlg.setFileMode(QFileDialog.ExistingFile)
            if self.exts:
                patt = " ".join(f"*{x}" for x in sorted(self.exts))
                dlg.setNameFilter(f"Supported files ({patt})")
            if dlg.exec():
                files = dlg.selectedFiles()
                if files and self._acceptable(files[0]): self.fileDropped.emit(files[0])

    def _acceptable(self, path: str) -> bool:
        if not os.path.isfile(path): return False
        if self.exts is None: return True
        return Path(path).suffix.lower() in self.exts

# ----------------- payload panel -----------------
class PayloadPanel(QWidget):
    def __init__(self, on_changed):
        super().__init__(); self.payload_path: str | None = None
        box = QVBoxLayout(self); self.tabs = QTabWidget()
        self.text_edit = QTextEdit(); self.text_edit.setPlaceholderText("Type or paste text payload here…")
        self.text_info = QLabel("Text bytes: 0"); self.text_edit.textChanged.connect(self._text_changed)
        t = QWidget(); tl = QVBoxLayout(t); tl.addWidget(self.text_edit); tl.addWidget(self.text_info)
        self.file_drop = DropLabel("a payload file (any type)", None); self.file_drop.fileDropped.connect(self._load_file)
        self.file_info = QLabel("No payload file loaded"); f = QWidget(); fl = QVBoxLayout(f); fl.addWidget(self.file_drop); fl.addWidget(self.file_info)
        self.tabs.addTab(t, "Text"); self.tabs.addTab(f, "File"); self.tabs.currentChanged.connect(on_changed)
        box.addWidget(self.tabs); self._on_changed = on_changed

    def mode(self) -> str: return "text" if self.tabs.currentIndex() == 0 else "file"

    def payload_bytes(self) -> bytes | None:
        if self.mode() == "text":
            b = self.text_edit.toPlainText().encode("utf-8"); return b if b else None
        if self.payload_path and os.path.isfile(self.payload_path):
            return open(self.payload_path, "rb").read()
        return None

    def payload_bits(self) -> int | None:
        if self.mode() == "text":
            n = len(self.text_edit.toPlainText().encode("utf-8")); return n*8 if n>0 else None
        if self.payload_path and os.path.isfile(self.payload_path):
            return os.path.getsize(self.payload_path)*8
        return None

    def _text_changed(self):
        n = len(self.text_edit.toPlainText().encode("utf-8")); self.text_info.setText(f"Text bytes: {n}"); self._on_changed()

    def _load_file(self, path: str):
        try:
            size = os.path.getsize(path); self.payload_path = path
            self.file_info.setText(f"Path: {path}\nSize: {human_bytes(size)}"); self._on_changed()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

# ----------------- MP3Stego backend -----------------
class Mp3StegoBackend:
    """
    Supports either:
      1) Single binary: mp3stego -E msg -P pass in.wav out.mp3 ; mp3stego -X -P pass in.mp3 out.bin
      2) Two binaries : encode    -E msg -P pass in.wav out.mp3 ; decode    -P pass in.mp3 out.bin
    Runs the tool inside a space-free temp dir and stages the required `tables/` folder.
    """
    CAP_RE = re.compile(r"You can hide roughly\s+(\d+)\s+bits", re.I)

    def __init__(self):
        self.cmd_embed, self.cmd_extract, self.mode = self._probe()

    # ---- discovery ----
    def _probe(self):
        if shutil.which("mp3stego"):
            return "mp3stego", "mp3stego", "single"
        if shutil.which("encode") and shutil.which("decode"):
            return "encode", "decode", "split"
        return None, None, None

    def available(self) -> bool:
        return self.mode is not None

    def _exe_dir(self) -> Path | None:
        exe = self.cmd_embed or self.cmd_extract
        if not exe:
            return None
        found = shutil.which(exe)
        return Path(found).parent if found else None

    # ---- temp dir & resources ----
    def _space_free_tmpdir(self) -> str:
        if sys.platform.startswith("win"):
            base = Path(os.environ.get("SystemDrive", "C:") + "\\mp3stego_tmp")
        else:
            base = Path("/tmp/mp3stego")
        base.mkdir(parents=True, exist_ok=True)
        return tempfile.mkdtemp(dir=str(base))

    def _safe_copy(self, src: str, dst_dir: str, suffix: str) -> str:
        dst = Path(dst_dir) / (uuid.uuid4().hex + suffix)
        shutil.copyfile(src, dst)
        return str(dst)

    def _stage_tables(self, dst_dir: str) -> bool:
        candidates: list[Path] = []
        d = self._exe_dir()
        if d:
            candidates.append(d / "tables")
            candidates.append(d.parent / "tables")
        for c in candidates:
            if c.is_dir():
                shutil.copytree(c, Path(dst_dir) / "tables", dirs_exist_ok=True)
                return True
        return False

    # ---- main ops ----
    def embed(self, wav_in: str, payload_bytes: bytes, passphrase: str, mp3_out: str):
        if not self.available():
            raise RuntimeError("MP3Stego not found. Install it and add to PATH (mp3stego or encode/decode).")

        tmpdir = self._space_free_tmpdir()
        try:
            self._stage_tables(tmpdir)
            wav_safe = self._safe_copy(wav_in, tmpdir, ".wav")
            msg_path = str(Path(tmpdir) / "msg.bin")
            Path(msg_path).write_bytes(payload_bytes)
            out_tmp_mp3 = str(Path(tmpdir) / "out.mp3")

            cmd = [self.cmd_embed, "-E", msg_path, "-P", passphrase, wav_safe, out_tmp_mp3]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
            if res.returncode != 0:
                raise RuntimeError(
                    f"MP3Stego encode failed (code {res.returncode}).\n"
                    f"CMD: {' '.join(cmd)}\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}"
                )
            shutil.copyfile(out_tmp_mp3, mp3_out)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def extract(self, mp3_in: str, passphrase: str) -> bytes:
        if not self.available():
            raise RuntimeError("MP3Stego not found. Install it and add to PATH (mp3stego or encode/decode).")

        tmpdir = self._space_free_tmpdir()
        try:
            self._stage_tables(tmpdir)
            mp3_safe = self._safe_copy(mp3_in, tmpdir, ".mp3")
            out_bin = str(Path(tmpdir) / "out.bin")

            cmd = ([self.cmd_extract, "-X", "-P", passphrase, mp3_safe, out_bin]
                   if self.mode == "single"
                   else [self.cmd_extract, "-P", passphrase, mp3_safe, out_bin])

            res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
            if res.returncode != 0:
                hint = ""
                if ("OpenTable" in (res.stderr or "")) and ("tables" in (res.stderr or "")):
                    hint = "\nHINT: mp3stego needs its 'tables' folder next to the executable. " \
                           "Make sure your install includes it (we try to stage it automatically)."
                raise RuntimeError(
                    f"MP3Stego decode failed (code {res.returncode}).\n"
                    f"CMD: {' '.join(cmd)}\nSTDERR:\n{res.stderr}\nSTDOUT:\n{res.stdout}{hint}"
                )
            return Path(out_bin).read_bytes()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def estimate_capacity_bits(self, wav_in: str) -> int | None:
        if not self.available():
            raise RuntimeError("MP3Stego not found. Install it and add to PATH (mp3stego or encode/decode).")
        tmpdir = self._space_free_tmpdir()
        try:
            self._stage_tables(tmpdir)
            wav_safe = self._safe_copy(wav_in, tmpdir, ".wav")
            big = Path(tmpdir) / "big.bin"
            big.write_bytes(b"\x00" * (50 * 1024 * 1024))
            out_tmp_mp3 = str(Path(tmpdir) / "probe.mp3")
            cmd = [self.cmd_embed, "-E", str(big), "-P", "captest", wav_safe, out_tmp_mp3]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=tmpdir)
            m = self.CAP_RE.search(res.stderr or "") or self.CAP_RE.search(res.stdout or "")
            return int(m.group(1)) if m else None
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

MP3STEGO = Mp3StegoBackend()

# ---- MP3Stego envelope (magic + meta + payload) ----
MP3_WRAP_MAGIC = b"STGM\x01"  # 'STGM' + version=1

def mp3_wrap_payload(mode: int, filename: str | None, payload: bytes) -> bytes:
    """
    MAGIC (5) | meta_len (4 LE) | data_len (4 LE) | META | PAYLOAD
    META uses the same TLV as build_meta/parse_meta.
    """
    meta = build_meta(mode, filename, None)
    return (
        MP3_WRAP_MAGIC +
        struct.pack("<I", len(meta)) +
        struct.pack("<I", len(payload)) +
        meta + payload
    )

def mp3_unwrap_payload(buf: bytes) -> tuple[dict, bytes] | None:
    """
    Find and extract (meta_dict, payload_bytes). Returns None if not present/invalid.
    Tolerant: searches for MAGIC anywhere (some builds may prepend bytes).
    """
    i = buf.find(MP3_WRAP_MAGIC)
    if i < 0 or len(buf) < i + len(MP3_WRAP_MAGIC) + 8:
        return None
    off = i + len(MP3_WRAP_MAGIC)
    try:
        meta_len = struct.unpack("<I", buf[off:off+4])[0]; off += 4
        data_len = struct.unpack("<I", buf[off:off+4])[0]; off += 4
    except struct.error:
        return None
    end = off + meta_len + data_len
    if end > len(buf):
        return None
    meta = parse_meta(buf[off:off+meta_len]); off += meta_len
    payload = buf[off:end]
    return meta, payload

# ----------------- AUDIO ENCODE TAB -----------------
class AudioEncodeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cover_path: str | None = None
        self.audio_info = None  # frames, channels, sampwidth, rate
        self.is_mp3_cover = False

        left = QVBoxLayout()
        cov_box = QGroupBox("Cover Audio (WAV, MP3, FLAC, OGG)")
        cv = QVBoxLayout()
        self.cover_drop = DropLabel("an audio file", SUPPORTED_AUDIO_EXTS)
        self.cover_drop.fileDropped.connect(self.load_cover)
        self.cover_info = QLabel("No audio loaded"); self.cover_info.setWordWrap(True)
        cv.addWidget(self.cover_drop); cv.addWidget(self.cover_info); cov_box.setLayout(cv)

        pay_box = QGroupBox("Payload (Text or File)")
        self.payload_panel = PayloadPanel(self.update_capacity_label)
        pay_v = QVBoxLayout(); pay_v.addWidget(self.payload_panel); pay_box.setLayout(pay_v)

        ctrl = QGroupBox("Embedding Controls")
        form = QFormLayout()
        self.lsb_slider = QSlider(Qt.Horizontal); self.lsb_slider.setRange(1,8); self.lsb_slider.setValue(1)
        self.lsb_slider.setTickInterval(1); self.lsb_slider.setSingleStep(1); self.lsb_slider.setTickPosition(QSlider.TicksBelow)
        self.lsb_value = QLabel("1")
        lsb_row = QHBoxLayout(); lsb_row.addWidget(self.lsb_slider,1); lsb_row.addWidget(self.lsb_value,0)
        lsb_widget = QWidget(); lsb_widget.setLayout(lsb_row)
        self.lsb_slider.valueChanged.connect(lambda v: self.lsb_value.setText(str(v)))
        self.lsb_slider.valueChanged.connect(self.update_capacity_label)
        self.key_edit = QLineEdit(); self.key_edit.setPlaceholderText("Enter numeric/passphrase key (required)")

        self.audio_start = QSpinBox(); self.audio_start.setRange(0,0)
        self.audio_len   = QSpinBox(); self.audio_len.setRange(1,1)
        self.audio_start.valueChanged.connect(self._start_changed)
        self.audio_len.valueChanged.connect(self.update_capacity_label)

        self.start_sec = QDoubleSpinBox(); self.start_sec.setDecimals(3); self.start_sec.setRange(0.0, 0.0)
        self.start_sec.setSingleStep(0.010); self.start_sec.setSuffix(" s"); self.start_sec.valueChanged.connect(self._start_sec_changed)
        self.len_sec   = QDoubleSpinBox(); self.len_sec.setDecimals(3); self.len_sec.setRange(0.001, 0.001)
        self.len_sec.setSingleStep(0.010); self.len_sec.setSuffix(" s"); self.len_sec.valueChanged.connect(self._len_sec_changed)

        form.addRow("Number of LSBs:", lsb_widget)
        form.addRow("Key:", self.key_edit)
        form.addRow("Start sample:", self.audio_start)
        form.addRow("Length (samples):", self.audio_len)
        form.addRow("Start (seconds):", self.start_sec)
        form.addRow("Length (seconds):", self.len_sec)
        ctrl.setLayout(form)

        # ---- MP3 handling toggle ----
        mp3_box = QGroupBox("MP3 Input Mode")
        mp3_lay = QVBoxLayout()
        self.chk_use_mp3stego = QCheckBox("Use MP3Stego for MP3 input (outputs a decodable MP3)")
        self.chk_use_mp3stego.setChecked(True)
        self.chk_use_mp3stego.toggled.connect(self._mp3_mode_changed)
        self.lbl_mp3_hint = QLabel("Note: When enabled on an MP3 cover, LSB/ROI controls are ignored.")
        self.lbl_mp3_hint.setWordWrap(True)
        mp3_lay.addWidget(self.chk_use_mp3stego); mp3_lay.addWidget(self.lbl_mp3_hint)
        mp3_box.setLayout(mp3_lay)
        mp3_box.setVisible(False)  # only show when the cover is actually an MP3

        ts_box = QGroupBox("Timestamp (slide to choose start)")
        tsv = QVBoxLayout(); row = QHBoxLayout()
        self.lbl_time_left = QLabel("0:00"); self.lbl_time_cur  = QLabel("0:00"); self.lbl_time_right = QLabel("0:00")
        self.lbl_time_cur.setAlignment(Qt.AlignCenter); self.lbl_time_cur.setStyleSheet("font-weight:600;")
        row.addWidget(self.lbl_time_left, 0); row.addWidget(self.lbl_time_cur, 1); row.addWidget(self.lbl_time_right, 0)
        tsv.addLayout(row); self.time_slider = QSlider(Qt.Horizontal); self.time_slider.setRange(0, 0)
        self.time_slider.setSingleStep(1); self.time_slider.setPageStep(44100)
        self.time_slider.valueChanged.connect(self._time_slider_changed)
        tsv.addWidget(self.time_slider); ts_box.setLayout(tsv)

        cap_box = QGroupBox("Capacity")
        self.cap_label = QLabel("Load WAV + set ROI + add payload."); self.cap_label.setWordWrap(True)
        cap_v = QVBoxLayout(); cap_v.addWidget(self.cap_label); cap_box.setLayout(cap_v)

        key_box = QGroupBox("Final Key (copy for decoding)")
        key_h = QHBoxLayout(); self.key_token_edit = QLineEdit(); self.key_token_edit.setReadOnly(True)
        self.copy_btn = QPushButton("Copy"); self.copy_btn.clicked.connect(self.copy_key_token)
        key_h.addWidget(self.key_token_edit); key_h.addWidget(self.copy_btn); key_box.setLayout(key_h)

        btns = QHBoxLayout(); self.encode_btn = QPushButton("Encode"); self.encode_btn.clicked.connect(self.on_encode); btns.addWidget(self.encode_btn)

        left.addWidget(cov_box); left.addWidget(pay_box); left.addWidget(ctrl)
        left.addWidget(mp3_box)  # new
        left.addWidget(ts_box)
        left.addWidget(cap_box); left.addWidget(key_box); left.addLayout(btns); left.addStretch(1)

        right = QVBoxLayout()
        wf_box = QGroupBox("Waveform (click–drag to select ROI)")
        wf_v = QVBoxLayout(); self.wave = WaveformView(); self.wave.selectionChanged.connect(self._on_wave_selection)
        wf_v.addWidget(self.wave); wf_box.setLayout(wf_v)
        log_box = QGroupBox("Log"); self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        lv = QVBoxLayout(); lv.addWidget(self.log_edit); log_box.setLayout(lv)
        right.addWidget(wf_box); right.addWidget(log_box)

        splitter = QSplitter(); lw = QWidget(); lw.setLayout(left); rw = QWidget(); rw.setLayout(right)
        splitter.addWidget(lw); splitter.addWidget(rw); splitter.setSizes([540, 680])
        main = QVBoxLayout(self); main.addWidget(splitter)

        # keep references to some widgets for enabling/disabling in MP3 modes
        self._ctrl_widgets_lsb = [self.lsb_slider, self.audio_start, self.audio_len, self.start_sec, self.len_sec, self.time_slider]
        self._mp3_mode_box = mp3_box

    def _set_lsb_controls_enabled(self, enabled: bool):
        for w in self._ctrl_widgets_lsb:
            w.setEnabled(enabled)

    def _mp3_mode_changed(self, _):
        # Only matters if current cover is MP3
        if not self.is_mp3_cover:
            return
        use_mp3stego = self.chk_use_mp3stego.isChecked()
        self._set_lsb_controls_enabled(not use_mp3stego)
        self.update_capacity_label()

    def copy_key_token(self):
        QtGui.QGuiApplication.clipboard().setText(self.key_token_edit.text())
        QMessageBox.information(self, "Copied", "Final Key copied to clipboard.")

    # WAV helpers (for LSB path)
    def _read_wav_mono(self, path: str):
        with wave_mod.open(path, "rb") as wf:
            n_channels = wf.getnchannels(); sampwidth = wf.getsampwidth()
            framerate  = wf.getframerate(); n_frames  = wf.getnframes(); raw = wf.readframes(n_frames)
        if sampwidth == 1: a = (np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) << 8
        elif sampwidth == 2: a = np.frombuffer(raw, dtype=np.int16)
        elif sampwidth == 3:
            b = np.frombuffer(raw, dtype=np.uint8); b = b[: (len(b)//3)*3].reshape(-1,3)
            a = (b[:,0].astype(np.int32) | (b[:,1].astype(np.int32)<<8) | (b[:,2].astype(np.int32)<<16))
            mask = 1 << 23; a = (a ^ mask) - mask
        elif sampwidth == 4: a = np.frombuffer(raw, dtype=np.int32)
        else: raise ValueError(f"Unsupported sample width: {sampwidth*8}-bit")
        if n_channels > 1: a = a.reshape(-1, n_channels).mean(axis=1).astype(a.dtype)
        info = {"frames": n_frames, "channels": n_channels, "sampwidth": sampwidth, "rate": framerate}
        return a, framerate, info

    def _read_wav_bytes(self, path: str):
        with wave_mod.open(path, "rb") as wf:
            n_channels = wf.getnchannels(); sampwidth = wf.getsampwidth()
            framerate  = wf.getframerate(); n_frames  = wf.getnframes(); raw = wf.readframes(n_frames)
        raw_u8 = np.frombuffer(raw, dtype=np.uint8).copy()
        params = {"channels": n_channels, "sampwidth": sampwidth, "rate": framerate, "frames": n_frames}
        return raw_u8, params

    def _write_wav_bytes(self, out_path: str, params: dict, raw_u8: np.ndarray):
        with wave_mod.open(out_path, "wb") as wf:
            wf.setnchannels(params["channels"]); wf.setsampwidth(params["sampwidth"])
            wf.setframerate(params["rate"]); wf.writeframes(raw_u8.tobytes())

    # --------- bit helpers (LSB) ----------
    def _target_byte_indices(self, params: dict, start_frame: int, length_frames: int) -> np.ndarray:
        C = params["channels"]; B = params["sampwidth"]; F = length_frames
        bytes_per_frame = C * B; base = start_frame * bytes_per_frame
        idx = np.empty(F * C, dtype=np.int64); k = 0
        for i in range(F):
            frame_base = base + i * bytes_per_frame
            for c in range(C): idx[k] = frame_base + c * B; k += 1
        return idx

    def _bit_chunks(self, data: bytes, lsb: int):
        cur = []
        for byte in data:
            for bitpos in range(7, -1, -1):
                cur.append((byte >> bitpos) & 1)
                if len(cur) == lsb:
                    yield cur; cur = []
        if cur:
            while len(cur) < lsb: cur.append(0)
            yield cur

    def _rotate_bits(self, bits: list[int], lsb: int, kbit_byte: int) -> int:
        if lsb <= 0: return 0
        off = (kbit_byte % lsb)
        if off: bits = bits[off:] + bits[:off]
        val = 0
        for b in bits: val = (val << 1) | (b & 1)
        return val

    def _embed_bits_into_bytes(self, raw_u8, target_idx, bit_groups_iter, lsb, kbit_byte):
        mask = (1 << lsb) - 1; mask_u8 = np.uint8(mask); invmask_u8 = np.uint8(~mask_u8)
        for i in range(len(target_idx)):
            try: bits = next(bit_groups_iter)
            except StopIteration: break
            v = self._rotate_bits(bits, lsb, kbit_byte)
            j = target_idx[i]; base = np.uint8(raw_u8[j]) & invmask_u8
            raw_u8[j] = np.uint8(base | np.uint8(v))

    # --------- UI interactions ----------
    def load_cover(self, path: str):
        try:
            if not os.path.isfile(path): raise ValueError("File not found.")
            pcm = decode_any_to_pcm(path, force_mono=True)
            self.orig_cover_path = path; self.cover_path = path; self.pcm_meta = pcm

            tmp_wav = str(Path(tempfile.gettempdir()) / f"{Path(path).stem}_{uuid.uuid4().hex}.wav")
            write_pcm_as_wav(tmp_wav, pcm)
            self.wav_for_embed = tmp_wav

            info = {"frames": len(pcm.samples_i16), "channels": 1, "sampwidth": 2, "rate": pcm.rate}
            self.audio_info = info
            lossy_note = " (lossy source)" if pcm.is_lossy else ""
            self.cover_info.setText(f"Path: {path}\nDecoded to WAV mono for view @ {info['rate']}Hz, 16-bit, frames={info['frames']}{lossy_note}")

            # MP3 mode UI
            ext = Path(path).suffix.lower()
            self.is_mp3_cover = (ext == ".mp3")
            self._mp3_mode_box.setVisible(self.is_mp3_cover)
            self._set_lsb_controls_enabled(not (self.is_mp3_cover and self.chk_use_mp3stego.isChecked()))

            self.audio_start.blockSignals(True); self.audio_len.blockSignals(True)
            self.audio_start.setRange(0, max(0, info["frames"] - 1)); self.audio_start.setValue(0)
            self.audio_len.setRange(1, info["frames"]); self.audio_len.setValue(info["frames"])
            self.audio_start.blockSignals(False); self.audio_len.blockSignals(False)

            total_dur = info["frames"] / info["rate"] if info["rate"] > 0 else 0.0
            min_dur   = 1.0 / info["rate"] if info["rate"] > 0 else 0.001
            self.start_sec.blockSignals(True); self.len_sec.blockSignals(True)
            self.start_sec.setRange(0.0, max(0.0, total_dur))
            self.len_sec.setRange(min_dur, max(min_dur, total_dur))
            self.start_sec.setValue(0.0); self.len_sec.setValue(max(min_dur, total_dur))
            self.start_sec.blockSignals(False); self.len_sec.blockSignals(False)

            self.time_slider.blockSignals(True); self.time_slider.setRange(0, max(0, info["frames"] - 1))
            self.time_slider.setPageStep(max(1, info["rate"])); self.time_slider.setValue(0); self.time_slider.blockSignals(False)

            self.lbl_time_left.setText("0:00"); self.lbl_time_right.setText(fmt_time(info["frames"], info["rate"])); self.lbl_time_cur.setText("0:00")

            mono = pcm.samples_i16 if pcm.samples_i16.ndim == 1 else pcm.samples_i16[:,0]
            self.wave.set_audio(mono, pcm.rate); self.wave.set_selection(self.audio_start.value(), self.audio_len.value())
            self.update_capacity_label()

            if pcm.is_lossy and not (self.is_mp3_cover and self.chk_use_mp3stego.isChecked()):
                self.log("[WARN] Source is lossy (e.g., MP3). LSB payload won’t survive re-encoding. "
                         "If you encode with LSB and later transcode to MP3, the payload will not decode.")
            if self.is_mp3_cover and self.chk_use_mp3stego.isChecked():
                self.log("[INFO] MP3 cover detected: MP3Stego mode selected. ROI/LSB controls are disabled.")
            self.log(f"Loaded audio: {path}")
        except Exception as e:
            self.error(str(e))

    def _on_wave_selection(self, start_sample: int, length: int):
        if self.audio_info:
            max_len = max(1, self.audio_info["frames"] - start_sample); length = max(1, min(length, max_len))
        self.audio_start.blockSignals(True); self.audio_len.blockSignals(True)
        self.audio_start.setValue(start_sample); self.audio_len.setValue(length)
        self.audio_start.blockSignals(False); self.audio_len.blockSignals(False)
        if self.audio_info:
            rate = self.audio_info["rate"]; self.start_sec.blockSignals(True); self.len_sec.blockSignals(True)
            self.start_sec.setValue(start_sample / rate); self.len_sec.setValue(length / rate)
            max_len_sec = (self.audio_info["frames"] - start_sample) / rate
            self.len_sec.setMaximum(max_len_sec if max_len_sec > 0 else self.len_sec.minimum())
            self.start_sec.blockSignals(False); self.len_sec.blockSignals(False)
        self._set_time_slider(start_sample); self.update_capacity_label()

    def _set_time_slider(self, start_sample: int):
        self.time_slider.blockSignals(True); self.time_slider.setValue(start_sample); self.time_slider.blockSignals(False)
        if self.audio_info: self.lbl_time_cur.setText(fmt_time(start_sample, self.audio_info["rate"]))

    def _time_slider_changed(self, val: int):
        if not self.audio_info: return
        self.lbl_time_cur.setText(fmt_time(val, self.audio_info["rate"]))
        self.audio_start.blockSignals(True); self.audio_start.setValue(val); self.audio_start.blockSignals(False)
        rate = self.audio_info["rate"]; self.start_sec.blockSignals(True); self.start_sec.setValue(val / rate); self.start_sec.blockSignals(False)
        self.wave.set_selection(self.audio_start.value(), self.audio_len.value()); self.update_capacity_label()

    def _start_changed(self, start_val: int):
        if self.audio_info:
            max_len = max(1, self.audio_info["frames"] - start_val); self.audio_len.setMaximum(max_len)
        self.wave.set_selection(self.audio_start.value(), self.audio_len.value()); self._set_time_slider(self.audio_start.value())
        if self.audio_info:
            rate = self.audio_info["rate"]; self.start_sec.blockSignals(True); self.len_sec.blockSignals(True)
            self.start_sec.setValue(self.audio_start.value() / rate); self.len_sec.setValue(self.audio_len.value() / rate)
            max_len_sec = (self.audio_info["frames"] - self.audio_start.value()) / rate
            self.len_sec.setMaximum(max_len_sec if max_len_sec > 0 else self.len_sec.minimum())
            self.start_sec.blockSignals(False); self.len_sec.blockSignals(False)
        self.update_capacity_label()

    def _start_sec_changed(self, s: float):
        if not self.audio_info: return
        rate = self.audio_info["rate"]; total = self.audio_info["frames"]
        start_f, len_f = secs_to_frames(s, self.len_sec.value(), rate, total)
        self.audio_start.blockSignals(True); self.audio_len.blockSignals(True)
        self.audio_start.setValue(start_f); self.audio_len.setValue(len_f)
        self.audio_start.blockSignals(False); self.audio_len.blockSignals(False)
        self._set_time_slider(start_f); self.wave.set_selection(start_f, len_f)
        max_len_sec = (total - start_f) / rate
        self.len_sec.blockSignals(True); self.len_sec.setMaximum(max_len_sec if max_len_sec > 0 else self.len_sec.minimum()); self.len_sec.blockSignals(False)
        self.update_capacity_label()

    def _len_sec_changed(self, d: float):
        if not self.audio_info: return
        rate = self.audio_info["rate"]; total = self.audio_info["frames"]
        start_f, len_f = secs_to_frames(self.start_sec.value(), d, rate, total)
        self.audio_len.blockSignals(True); self.audio_len.setValue(len_f); self.audio_len.blockSignals(False)
        self.wave.set_selection(start_f, len_f); self.update_capacity_label()

    def current_lsb(self) -> int: return self.lsb_slider.value()

    def update_capacity_label(self):
        if not self.cover_path or not self.audio_info:
            self.cap_label.setText("Load WAV + set ROI + add payload."); return

        if self.is_mp3_cover and self.chk_use_mp3stego.isChecked():
            # MP3Stego mode
            self.cap_label.setText(
                "MP3Stego mode selected.\n"
                "Capacity depends on MP3 coding and content; the tool will refuse if the message is too large.\n"
                "ROI/LSB controls are ignored in this mode."
            )
            return

        # LSB capacity (works for WAV/FLAC or MP3-to-WAV path)
        lsb = self.current_lsb(); start = self.audio_start.value(); length = self.audio_len.value()
        length = max(1, min(length, self.audio_info["frames"] - start))
        channels = self.audio_info["channels"]; capacity_bits = length * channels * lsb
        payload_bits = self.payload_panel.payload_bits()
        if payload_bits is None:
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\n"
                f"Start {self.start_sec.value():.3f}s, Length {self.len_sec.value():.3f}s\n"
                "Add payload (Text or File)."
            )
        else:
            ok = payload_bits <= capacity_bits
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\n"
                f"Start {self.start_sec.value():.3f}s, Length {self.len_sec.value():.3f}s\n"
                f"Payload: {payload_bits} bits ({human_bytes(payload_bits//8)})\n"
                f"Result: {'OK' if ok else 'Payload too large, require more capacity'}"
            )

    def _prep_wav_for_mp3stego(self) -> str:
        """Convert original cover to 44.1kHz, 16-bit, stereo WAV that MP3Stego expects."""
        a = AudioSegment.from_file(self.orig_cover_path)
        a = a.set_channels(2).set_frame_rate(44100).set_sample_width(2)  # stereo, 16-bit, 44.1k
        tmp = Path(tempfile.gettempdir()) / f"mp3stego_in_{uuid.uuid4().hex}.wav"
        a.export(tmp, format="wav")
        return str(tmp)

    def _export_mp3_preview_from_wav(self, wav_path: str, suggested_name_stem: str) -> str:
        """Produce a non-decodable MP3 from a stego WAV (for A/B listening)."""
        out_mp3 = SAFE_OUTDIR / f"{suggested_name_stem}_stego_preview.mp3"
        a = AudioSegment.from_wav(wav_path)
        a.export(out_mp3, format="mp3")  # uses system ffmpeg
        return str(out_mp3)

    # --------------------- ENCODE (routes by extension) ---------------------
    def on_encode(self):
        if not self.cover_path or not self.audio_info:
            self.error("Load a cover audio file."); return
        payload = self.payload_panel.payload_bytes()
        if not payload: self.error("Enter payload text or choose a payload file."); return
        key = self.key_edit.text().strip()
        if not key: self.error("Key is required."); return

        ext = Path(self.orig_cover_path).suffix.lower()
        is_mp3 = (ext == ".mp3")
        use_mp3stego = (is_mp3 and self.chk_use_mp3stego.isChecked())

        if is_mp3 and use_mp3stego:
            # --- MP3 path (robust, decodable) ---
            try:
                wav_for_mp3stego = self._prep_wav_for_mp3stego()
                out_mp3 = SAFE_OUTDIR / f"{Path(self.orig_cover_path).stem}_stego.mp3"

                # Wrap so we can recover filename/type and exact length later
                mode = 0 if self.payload_panel.mode() == "text" else 1
                fname = (os.path.basename(self.payload_panel.payload_path)
                        if (mode == 1 and self.payload_panel.payload_path) else None)
                payload_wrapped = mp3_wrap_payload(mode, fname, payload)

                MP3STEGO.embed(wav_for_mp3stego, payload_wrapped, key, str(out_mp3))
                self.key_token_edit.setText("(mp3stego)")  # MP3Stego uses the passphrase only
                self.log(f"[MP3Stego] Saved stego MP3: {out_mp3}")
                QMessageBox.information(self, "Encode complete", f"Stego MP3 written:\n{out_mp3}")
            except Exception as e:
                self.error(f"MP3 stego failed: {e}")
            return

        # --- WAV/FLAC path (PCM LSB) --- (also used when MP3 cover but user selects WAV-LSB)
        try:
            lsb = self.current_lsb(); start = self.audio_start.value()
            length = max(1, min(self.audio_len.value(), self.audio_info["frames"] - start))
            roi = (start, 0, length, 0)
            cover_id = cover_fingerprint(getattr(self, "orig_cover_path", self.cover_path))
            full_salt = canonical_salt(lsb, roi, cover_id, "audio"); salt16 = full_salt[:16]
            kd = kdf_from_key(key, salt16)
            K_perm, K_bit, K_check, nonce = kd["K_perm"], kd["K_bit"], kd["K_check"], kd["nonce"]

            mode = 0 if self.payload_panel.mode() == "text" else 1
            fname = os.path.basename(self.payload_panel.payload_path) if (mode == 1 and self.payload_panel.payload_path) else None
            meta = build_meta(mode, fname, None); meta_len_bytes = struct.pack("<I", len(meta))

            header = build_header(1, lsb, roi, len(payload), cover_id, salt16, nonce, K_check)
            token  = make_key_token("audio", lsb, roi, salt16, K_check); self.key_token_edit.setText(token)

            raw_u8, params = self._read_wav_bytes(self.wav_for_embed)
            tgt = self._target_byte_indices(params, start, length)
            total_bits = (len(header) + 4 + len(meta) + len(payload)) * 8
            capacity_bits = len(tgt) * lsb
            if total_bits > capacity_bits:
                self.error(f"Not enough capacity in ROI.\nNeed {total_bits} bits, have {capacity_bits} bits."); return

            rng_seed = int.from_bytes(K_perm, "little", signed=False)
            rng = Generator(PCG64(rng_seed)); perm = rng.permutation(len(tgt)); tgt_perm = tgt[perm]
            bit_groups = self._bit_chunks(header + meta_len_bytes + meta + payload, lsb)
            kbit_byte = K_bit[0]; used_targets = int(math.ceil(total_bits / lsb))
            impacted_byte_idx = tgt_perm[:used_targets]
            C = params["channels"]; B = params["sampwidth"]; bytes_per_frame = C * B
            impacted_frames = (impacted_byte_idx // bytes_per_frame).astype(np.int64)
            self.wave.highlight_impacts(impacted_frames, params["rate"])
            self._embed_bits_into_bytes(raw_u8, tgt_perm, bit_groups, lsb, kbit_byte)

            out_wav = SAFE_OUTDIR / f"{Path(self.orig_cover_path).stem}_stego.wav"
            if getattr(self, "pcm_meta", None) and self.pcm_meta.is_lossy:
                self.log("[NOTE] Stego written as WAV because LSB requires lossless PCM. "
                        "Do NOT convert this stego.wav to MP3 if you intend to decode later.")
            self._write_wav_bytes(str(out_wav), params, raw_u8)

            # waveform comparison
            orig_samples, rate, _ = self._read_wav_mono(self.wav_for_embed)
            stego_samples, _, _   = self._read_wav_mono(str(out_wav))
            self.wave.set_audio(stego_samples, rate); self.wave.set_selection(self.audio_start.value(), self.audio_len.value())
            if hasattr(self.wave, "set_comparison"):
                self.wave.set_comparison(orig_samples, stego_samples); self.wave.highlight_impacts(impacted_frames, params["rate"])
            diff = np.abs(orig_samples.astype(np.int32) - stego_samples.astype(np.int32))
            self.log(f"Waveform comparison: max diff={int(diff.max())}, mean diff={float(diff.mean()):.2f}")

            self.log(f"Derived token: {token}")
            self.log(f"Embedded {total_bits} bits into {len(tgt)} target bytes @ {lsb} LSB(s).")
            self.log(f"Saved stego audio: {out_wav}")

            # Optional preview MP3 for A/B if original was MP3 but user chose LSB
            if getattr(self, "is_mp3_cover", False) and not self.chk_use_mp3stego.isChecked():
                preview_mp3 = self._export_mp3_preview_from_wav(str(out_wav), Path(self.orig_cover_path).stem)
                self.log(f"[Preview] Also wrote non-decodable MP3 for listening: {preview_mp3}")

            QMessageBox.information(
                self, "Encode complete",
                f"Stego written:\n{out_wav}"
                + ("\n\nAlso wrote a non-decodable MP3 preview." if getattr(self, "is_mp3_cover", False) and not self.chk_use_mp3stego.isChecked() else "")
                + "\n\nCopy the Final Key for decoding."
            )
        except Exception as e:
            self.error(str(e))

    def log(self, msg: str): self.log_edit.append(msg)
    def error(self, msg: str): QMessageBox.critical(self, "Error", msg); self.log(f"[ERROR] {msg}")

# ----------------- AUDIO DECODE TAB -----------------
class AudioDecodeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.stego_path: str | None = None
        root = QVBoxLayout(self)
        media_box = QGroupBox("Stego Audio (WAV/FLAC/OGG or MP3+ID3)")
        mv = QVBoxLayout()
        self.stego_drop = DropLabel("a stego audio file", SUPPORTED_AUDIO_EXTS)
        self.stego_drop.fileDropped.connect(self.load_stego)
        self.media_info = QLabel("No stego audio loaded"); self.media_info.setWordWrap(True)
        mv.addWidget(self.stego_drop); mv.addWidget(self.media_info); media_box.setLayout(mv)

        ctrl = QGroupBox("Controls")
        form = QFormLayout()
        self.key_token_edit = QLineEdit(); self.key_token_edit.setPlaceholderText("Paste Final Key (LSB path) or leave blank for MP3Stego")
        self.user_key_edit = QLineEdit(); self.user_key_edit.setPlaceholderText("Enter the same key / passphrase")
        form.addRow("Final Key:", self.key_token_edit); form.addRow("User Key:", self.user_key_edit)
        ctrl.setLayout(form)

        btns = QHBoxLayout(); self.inspect_btn = QPushButton("Inspect Header / Presence"); self.decode_btn  = QPushButton("Decode Payload")
        self.inspect_btn.clicked.connect(self.on_inspect); self.decode_btn.clicked.connect(self.on_decode)
        btns.addWidget(self.inspect_btn); btns.addWidget(self.decode_btn)

        log_box = QGroupBox("Log"); self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        lv = QVBoxLayout(); lv.addWidget(self.log_edit); log_box.setLayout(lv)

        root.addWidget(media_box); root.addWidget(ctrl); root.addLayout(btns); root.addWidget(log_box)

    # WAV helpers
    def _read_wav_bytes(self, path: str):
        with wave_mod.open(path, "rb") as wf:
            n_channels = wf.getnchannels(); sampwidth = wf.getsampwidth()
            framerate  = wf.getframerate(); n_frames  = wf.getnframes(); raw = wf.readframes(n_frames)
        raw_u8 = np.frombuffer(raw, dtype=np.uint8).copy()
        params = {"channels": n_channels, "sampwidth": sampwidth, "rate": framerate, "frames": n_frames}
        return raw_u8, params

    def _target_byte_indices(self, params: dict, start_frame: int, length_frames: int) -> np.ndarray:
        C = params["channels"]; B = params["sampwidth"]; F = length_frames
        bytes_per_frame = C * B; base = start_frame * bytes_per_frame
        idx = np.empty(F * C, dtype=np.int64); k = 0
        for i in range(F):
            frame_base = base + i * bytes_per_frame
            for c in range(C): idx[k] = frame_base + c * B; k += 1
        return idx

    # Extraction helpers
    @staticmethod
    def _bits_from_byte(value_u8: int, lsb: int) -> list[int]:
        field = value_u8 & ((1 << lsb) - 1); out = []
        for i in range(lsb-1, -1, -1): out.append((field >> i) & 1)
        return out

    @staticmethod
    def _rotate_right(bits: list[int], lsb: int, off: int) -> list[int]:
        if lsb <= 0: return bits
        r = off % lsb
        if r == 0: return bits
        return bits[-r:] + bits[:-r]

    @staticmethod
    def _bits_to_bytes(msb_bits: list[int]) -> bytes:
        out = bytearray(); i = 0; n = len(msb_bits)
        while i < n:
            val = 0
            for j in range(8):
                val = (val << 1) | (msb_bits[i+j] if (i+j) < n else 0)
            out.append(val); i += 8
        return bytes(out)

    def load_stego(self, path: str):
        try:
            ext = Path(path).suffix.lower()
            self.stego_path = path
            if ext == ".wav":
                with wave_mod.open(path, "rb") as wf:
                    n_channels = wf.getnchannels(); sampwidth = wf.getsampwidth(); framerate = wf.getframerate(); n_frames  = wf.getnframes()
                self.media_info.setText(f"Path: {path}\nWAV {n_channels}ch @ {framerate}Hz, {sampwidth*8}-bit, frames={n_frames}")
            else:
                self.media_info.setText(f"Path: {path}")
            self.log(f"Loaded stego file: {path}")
        except Exception as e:
            self.error(str(e))

    def _extract_bits(self, raw_u8, tgt_indices, total_bits_needed, lsb, kbit_byte):
        bits_out = []; rotate = kbit_byte % lsb if lsb > 0 else 0
        for j in tgt_indices:
            field_bits = self._bits_from_byte(int(raw_u8[j]), lsb)
            if rotate: field_bits = self._rotate_right(field_bits, lsb, rotate)
            bits_out.extend(field_bits)
            if len(bits_out) >= total_bits_needed: return bits_out[:total_bits_needed]
        return bits_out

    def _rebuild_indices_perm(self, params, roi, K_perm):
        x0, _, w, _ = roi; tgt = self._target_byte_indices(params, x0, w)
        rng_seed = int.from_bytes(K_perm, "little", signed=False); rng = Generator(PCG64(rng_seed))
        perm = rng.permutation(len(tgt)); return tgt[perm]

    def _derive_keys_from_token(self, token: str, user_key: str):
        info = parse_key_token(token); kd = kdf_from_key(user_key, info["salt16"])
        if info["kcheck4"] != kd["K_check"]: raise ValueError("Wrong passphrase or tampered Final Key (K_check mismatch)")
        return info, kd

    def on_inspect(self):
        if not self.stego_path:
            self.error("Load a stego audio file first."); return
        ext = Path(self.stego_path).suffix.lower()
        if ext != ".wav":
            self.log("[MP3 path] Presence check is not supported here. Use Decode to extract if MP3Stego was used.")
            QMessageBox.information(self, "Inspect", "For MP3 stego, try Decode directly.")
            return
        token = self.key_token_edit.text().strip(); user_key = self.user_key_edit.text().strip()
        if not token or not user_key:
            self.error("Provide both Final Key token and the user key (WAV LSB path)."); return
        try:
            info, kd = self._derive_keys_from_token(token, user_key)
            raw_u8, params = self._read_wav_bytes(self.stego_path)
            lsb = info["lsb"]; roi = info["roi"]; K_perm = kd["K_perm"]; K_bit = kd["K_bit"]; K_check = kd["K_check"]
            tgt_perm = self._rebuild_indices_perm(params, roi, K_perm)
            capacity_bits = len(tgt_perm) * lsb
            if capacity_bits < HEADER_TOTAL_LEN*8: raise ValueError("ROI too small to contain header")
            header_bits  = self._extract_bits(raw_u8, tgt_perm, HEADER_TOTAL_LEN*8, lsb, K_bit[0])
            header_bytes = self._bits_to_bytes(header_bits); ph = parse_header(header_bytes)
            if ph.kcheck4 != K_check: raise ValueError("Wrong key/token (K_check mismatch)")
            ok_ctx = (ph.lsb == lsb and ph.roi == roi)
            self.log(f"Header OK: version={ph.version} payload_len={ph.payload_len} lsb={ph.lsb} roi={ph.roi} salt16={ph.salt16.hex()} crc32=0x{ph.crc32:08x}")
            self.log("Context matches token: " + ("YES" if ok_ctx else "NO (mismatch)"))
            QMessageBox.information(self, "Inspect", "Header parsed successfully. See log for details.")
        except Exception as e:
            self.error(str(e))

    def _save_with_dialog(self, suggested_name: str, data: bytes, text_mode: bool = False) -> None:
        suggested_name = _safe_filename(suggested_name, "payload.bin")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save payload as…", str(SAFE_OUTDIR / suggested_name), "All files (*)"
        )
        if not path:
            self.log("[Decode] Save cancelled by user.")
            return
        try:
            if text_mode:
                Path(path).write_text(data.decode("utf-8"), encoding="utf-8")
            else:
                Path(path).write_bytes(data)
            self.log(f"[Decode] Saved -> {path} ({len(data)} bytes)")
            QMessageBox.information(self, "Decode", f"Saved to:\n{path}")
        except Exception as e:
            self.error(f"Failed to save payload: {e}")

    def _looks_like_text(self, b: bytes) -> bool:
        try:
            s = b.decode("utf-8")
        except UnicodeDecodeError:
            return False
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in s)
        return printable / max(1, len(s)) > 0.95

    def _show_text_preview(self, text: str, byte_len: int):
        MAX_CHARS = 8000
        if len(text) <= MAX_CHARS:
            self.log(f"[Decode] Text payload ({byte_len} bytes):\n{text}")
        else:
            preview = text[:MAX_CHARS]
            self.log(f"[Decode] Text payload ({byte_len} bytes) — showing first {MAX_CHARS} chars:\n{preview}\n… (truncated)")
            QMessageBox.information(self, "Text payload truncated",
                                    f"The text is long ({byte_len} bytes). Only the first {MAX_CHARS} characters are shown in the log.")

    def on_decode(self):
        if not self.stego_path:
            self.error("Load a stego audio file first."); return

        ext = Path(self.stego_path).suffix.lower()
        user_key = self.user_key_edit.text().strip()
        if not user_key:
            self.error("Enter the key/passphrase."); return

        # --- helper: try to guess a cover sibling (optional) ---
        def _guess_cover_for(stego_path: str) -> str | None:
            p = Path(stego_path)
            # common pattern we created on encode: *_stego.*  -> try to find original stem
            stem = p.stem
            base = stem.replace("_stego", "")
            for sx in (".wav", ".flac", ".ogg", ".mp3"):
                cand = p.with_name(base + sx)
                if cand.is_file() and str(cand) != stego_path:
                    return str(cand)
            return None

        # ========== MP3 route (MP3Stego) ==========
        if ext == ".mp3":
            try:
                payload_bytes = MP3STEGO.extract(self.stego_path, user_key)
                self.log(f"[MP3Stego] Extracted {len(payload_bytes)} bytes; head={payload_bytes[:16].hex()}")

                # Prefer our envelope (supports binary + filename)
                unwrapped = mp3_unwrap_payload(payload_bytes)
                if unwrapped is not None:
                    meta, data = unwrapped
                    mode = int(meta.get("mode", 0))   # 0=text, 1=binary
                    fname = meta.get("filename") or ("payload.txt" if mode == 0 else "payload.bin")

                    dlg = AudioDecodePreview(
                        self,
                        stego_path=self.stego_path,
                        cover_path=_guess_cover_for(self.stego_path),
                        payload_bytes=data,
                        payload_name=fname,
                        payload_mode=mode
                    )
                    dlg.exec()
                else:
                    # Legacy/foreign MP3Stego: no envelope
                    mode_guess = 0 if self._looks_like_text(payload_bytes) else 1
                    fname = "payload.txt" if mode_guess == 0 else "payload.bin"
                    dlg = AudioDecodePreview(
                        self,
                        stego_path=self.stego_path,
                        cover_path=_guess_cover_for(self.stego_path),
                        payload_bytes=payload_bytes,
                        payload_name=fname,
                        payload_mode=mode_guess
                    )
                    dlg.exec()
            except Exception as e:
                self.error(f"MP3 stego decode failed: {e}")
            return

        # ========== WAV/FLAC/OGG route (LSB) ==========
        token = self.key_token_edit.text().strip()
        if not token:
            self.error("Final Key token is required for WAV/FLAC LSB decode."); return
        try:
            info, kd = self._derive_keys_from_token(token, user_key)
            if info["media_kind"] != "audio": raise ValueError("This token is not for audio")

            raw_u8, params = self._read_wav_bytes(self.stego_path)
            lsb = info["lsb"]; roi = info["roi"]; K_perm = kd["K_perm"]; K_bit = kd["K_bit"]; K_check = kd["K_check"]
            tgt_perm = self._rebuild_indices_perm(params, roi, K_perm)
            capacity_bits = len(tgt_perm) * lsb
            if capacity_bits < HEADER_TOTAL_LEN*8: raise ValueError("ROI too small to contain header")

            # header
            header_bits  = self._extract_bits(raw_u8, tgt_perm, HEADER_TOTAL_LEN*8, lsb, K_bit[0])
            header_bytes = self._bits_to_bytes(header_bits); ph = parse_header(header_bytes)
            if ph.kcheck4 != K_check: raise ValueError("Wrong key/token (K_check mismatch)")
            if ph.lsb != lsb or ph.roi != roi: raise ValueError("Token/controls do not match embedded header")

            # meta length, then meta + payload
            need_bits_meta_len   = 4 * 8
            total_needed_so_far  = HEADER_TOTAL_LEN*8 + need_bits_meta_len
            all_bits = self._extract_bits(raw_u8, tgt_perm, total_needed_so_far, lsb, K_bit[0])
            meta_len_bytes = self._bits_to_bytes(all_bits[HEADER_TOTAL_LEN*8 : total_needed_so_far])
            (meta_len,) = struct.unpack("<I", meta_len_bytes)

            need_bits_rest = (meta_len + ph.payload_len) * 8
            total_needed   = total_needed_so_far + need_bits_rest
            all_bits = self._extract_bits(raw_u8, tgt_perm, total_needed, lsb, K_bit[0])

            offset      = HEADER_TOTAL_LEN*8 + need_bits_meta_len
            meta_bits   = all_bits[offset : offset + meta_len*8]
            payload_bits= all_bits[offset + meta_len*8 : offset + meta_len*8 + ph.payload_len*8]

            meta          = parse_meta(self._bits_to_bytes(meta_bits))
            payload_bytes = self._bits_to_bytes(payload_bits)

            mode  = int(meta.get("mode", 0))
            fname = meta.get("filename") or ("payload.txt" if mode == 0 else "payload.bin")

            # Show preview dialog (user can save from there)
            dlg = AudioDecodePreview(
                self,
                stego_path=self.stego_path,
                cover_path=None,  # unknown; user can load via the dialog button
                payload_bytes=payload_bytes,
                payload_name=fname,
                payload_mode=mode
            )
            dlg.exec()
        except Exception as e:
            self.error(str(e))

    def log(self, msg: str):
        self.log_edit.append(msg)

    def error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.log(f"[ERROR] {msg}")


class AudioDecodePreview(QtWidgets.QDialog):
    """
    Stego (left) vs Cover (right) waveforms, linked zoom/pan.
    Payload preview at the bottom: shows text or binary summary + Save.
    """
    def __init__(self, parent=None, stego_path:str|Path=None, cover_path:str|Path|None=None,
                 payload_bytes:bytes|None=None, payload_name:str|None=None, payload_mode:int=1):
        super().__init__(parent)
        self.setWindowTitle("Decode Preview")
        self.resize(1100, 700)

        self.stego_path = str(stego_path) if stego_path else None
        self.cover_path = str(cover_path) if cover_path else None
        self.payload_bytes = payload_bytes or b""
        self.payload_mode = int(payload_mode)  # 0=text, 1=binary
        self.payload_name = payload_name or ("payload.txt" if self.payload_mode == 0 else "payload.bin")

        main = QVBoxLayout(self)

        # --- Top: two waveforms
        top = QSplitter(Qt.Horizontal)
        self._left = self._build_wave_panel("Stego")
        self._right = self._build_wave_panel("Cover")
        top.addWidget(self._left["box"])
        top.addWidget(self._right["box"])
        top.setSizes([550, 550])
        main.addWidget(top, 2)

        # toolbar row (fit/zoom and spectrogram toggle)
        tools = QHBoxLayout()
        self.btn_fit = QPushButton("Fit")
        self.btn_fit.clicked.connect(self._fit_both)
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_out = QPushButton("–")
        self.btn_zoom_in.clicked.connect(lambda: self._zoom(0.8))
        self.btn_zoom_out.clicked.connect(lambda: self._zoom(1.25))
        self.chk_link = QtWidgets.QCheckBox("Link zoom/pan"); self.chk_link.setChecked(True)
        self.chk_spec = QtWidgets.QCheckBox("Spectrogram"); self.chk_spec.setChecked(False)
        self.chk_spec.stateChanged.connect(self._replot_both)
        tools.addWidget(self.btn_fit); tools.addWidget(self.btn_zoom_in); tools.addWidget(self.btn_zoom_out)
        tools.addSpacing(12); tools.addWidget(self.chk_link); tools.addSpacing(12); tools.addWidget(self.chk_spec)
        tools.addStretch(1)

        # Optional: button to load a cover file if none supplied
        self.btn_load_cover = QPushButton("Load cover for compare…")
        self.btn_load_cover.clicked.connect(self._choose_cover)
        tools.addWidget(self.btn_load_cover)
        main.addLayout(tools)

        # --- Bottom: payload preview
        payload_box = QGroupBox("Payload")
        pv = QVBoxLayout(payload_box)

        self.lbl_meta = QLabel("")
        pv.addWidget(self.lbl_meta)

        self.text_view = QTextEdit(); self.text_view.setReadOnly(True); self.text_view.hide()
        pv.addWidget(self.text_view, 1)

        row_btns = QHBoxLayout()
        self.btn_save_payload = QPushButton("Save payload…")
        self.btn_open_folder = QPushButton("Open payload folder")
        self.btn_save_payload.clicked.connect(self._save_payload_dialog)
        self.btn_open_folder.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(SAFE_OUTDIR))))
        row_btns.addStretch(1); row_btns.addWidget(self.btn_save_payload); row_btns.addWidget(self.btn_open_folder)
        pv.addLayout(row_btns)

        main.addWidget(payload_box, 1)

        # Load the audio now
        self._load_audio()
        self._populate_payload()

    # ---------- UI builders ----------
    def _build_wave_panel(self, title: str):
        box = QGroupBox(title)
        v = QVBoxLayout(box)
        fig = Figure(figsize=(5, 2.5), facecolor="white")
        canvas = FigureCanvasQTAgg(fig)
        ax = fig.add_subplot(111)
        ax.set_facecolor("#f9fbff")
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        v.addWidget(canvas)

        # status under plot
        status = QLabel(" — ")
        v.addWidget(status)

        # link pan/zoom
        def on_xlims(ax_):
            if not self.chk_link.isChecked():
                return
            other = self._right if ax_ is self._left["ax"] else self._left
            try:
                other["ax"].set_xlim(ax_.get_xlim())
                other["canvas"].draw_idle()
            except Exception:
                pass

        # Matplotlib callbacks
        def _on_xlims_changed(ax_):
            on_xlims(ax_)

        ax.callbacks.connect("xlim_changed", _on_xlims_changed)

        return {"box": box, "fig": fig, "canvas": canvas, "ax": ax, "status": status, "data": None}

    # ---------- Audio & plotting ----------
    def _pcm_from_path(self, path:str) -> tuple[np.ndarray,int]:
        pcm = decode_any_to_pcm(path, force_mono=True)
        mono = pcm.samples_i16 if pcm.samples_i16.ndim == 1 else pcm.samples_i16[:,0]
        return mono.astype(np.int32), int(pcm.rate)

    def _plot_wave(self, panel, samples: np.ndarray, rate: int, title: str):
        ax = panel["ax"]; ax.cla()
        ax.set_facecolor("#f9fbff"); ax.grid(True, alpha=0.25)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
        if self.chk_spec.isChecked():
            # quick&clean spectrogram
            from matplotlib.colors import LogNorm
            NFFT = 1024; nover = 768
            ax.specgram(samples.astype(np.float32)/32768.0, NFFT=NFFT, Fs=rate, noverlap=nover, cmap="magma", scale="dB")
            ax.set_ylabel("Freq (Hz)")
        else:
            n = len(samples); step = max(1, n // 200_000)
            y = (samples[::step].astype(np.float32))
            m = float(np.max(np.abs(y))) or 1.0
            y = y / m
            x = (np.arange(len(y)) * step) / rate
            ax.plot(x, y, linewidth=1.0)
            ax.set_ylim(-1.1, 1.1)
        ax.set_title(title)
        panel["canvas"].draw_idle()

    def _load_audio(self):
        # left: stego
        if self.stego_path and os.path.isfile(self.stego_path):
            s, r = self._pcm_from_path(self.stego_path)
            self._left["data"] = (s, r)
            self._plot_wave(self._left, s, r, Path(self.stego_path).name)
            self._left["status"].setText(f"{len(s)/max(1,r):.2f}s @ {r} Hz  |  frames={len(s)}")

        # right: cover (optional)
        if self.cover_path and os.path.isfile(self.cover_path):
            c, rc = self._pcm_from_path(self.cover_path)
            self._right["data"] = (c, rc)
            self._plot_wave(self._right, c, rc, Path(self.cover_path).name)
            self._right["status"].setText(f"{len(c)/max(1,rc):.2f}s @ {rc} Hz  |  frames={len(c)}")

            # diff metrics (only if both exist and same rate)
            if self._left["data"] and self._right["data"] and (self._left["data"][1] == rc):
                s = self._left["data"][0]; n = min(len(s), len(c))
                if n > 0:
                    d = (s[:n].astype(np.int32) - c[:n].astype(np.int32))
                    mx = int(np.max(np.abs(d)))
                    mse = float(np.mean((d.astype(np.float64))**2))
                    snr = 10.0 * math.log10((np.mean((c[:n].astype(np.float64))**2)+1e-12) / (mse+1e-12))
                    self._left["status"].setText(self._left["status"].text() + f"    |    maxΔ={mx}  SNR≈{snr:.2f} dB")

    def _replot_both(self):
        if self._left["data"]:
            s, r = self._left["data"]; self._plot_wave(self._left, s, r, Path(self.stego_path).name)
        if self._right["data"]:
            c, rc = self._right["data"]; self._plot_wave(self._right, c, rc, Path(self.cover_path).name)

    def _fit_both(self):
        for p in (self._left, self._right):
            try:
                ax = p["ax"]
                x0, x1 = 0.0, ax.lines[0].get_xdata()[-1] if ax.lines else ax.get_xlim()[1]
                ax.set_xlim(x0, x1)
                p["canvas"].draw_idle()
            except Exception:
                pass

    def _zoom(self, factor: float):
        for p in (self._left, self._right):
            try:
                ax = p["ax"]; x0, x1 = ax.get_xlim()
                cx = 0.5*(x0+x1); hw = 0.5*(x1-x0)*factor
                ax.set_xlim(max(0.0, cx-hw), max(cx+hw, 1e-3))
                p["canvas"].draw_idle()
            except Exception:
                pass

    def _choose_cover(self):
        dlg = QFileDialog(self); dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Audio files (*.wav *.mp3 *.flac *.ogg)")
        if dlg.exec():
            f = dlg.selectedFiles()[0]
            self.cover_path = f
            self._right["data"] = None
            self._load_audio()

    # ---------- Payload preview ----------
    def _looks_like_text(self, b: bytes) -> bool:
        try:
            s = b.decode("utf-8"); 
        except UnicodeDecodeError:
            return False
        printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in s)
        return printable / max(1, len(s)) > 0.95

    def _populate_payload(self):
        n = len(self.payload_bytes)
        if self.payload_mode == 0 and self._looks_like_text(self.payload_bytes):
            self.text_view.setPlainText(self.payload_bytes.decode("utf-8"))
            self.text_view.show()
            self.lbl_meta.setText(f"{self.payload_name}  •  {n} bytes  •  UTF-8 text")
        else:
            self.text_view.hide()
            self.lbl_meta.setText(f"{self.payload_name}  •  {n} bytes  •  binary")

    def _save_payload_dialog(self):
        suggested = _safe_filename(self.payload_name, "payload.bin" if self.payload_mode else "payload.txt")
        path, _ = QFileDialog.getSaveFileName(self, "Save payload as…", str(SAFE_OUTDIR / suggested), "All files (*)")
        if not path:
            return
        try:
            # If it is text, save as UTF-8
            if self.payload_mode == 0 and self._looks_like_text(self.payload_bytes):
                Path(path).write_text(self.payload_bytes.decode("utf-8"), encoding="utf-8")
            else:
                Path(path).write_bytes(self.payload_bytes)
            QMessageBox.information(self, "Saved", f"Saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save payload: {e}")


# -------------------------------
# Suite
class AudioSuite(QWidget):
    def __init__(self):
        super().__init__()
        tabs = QTabWidget(); tabs.addTab(AudioEncodeTab(), "Encode"); tabs.addTab(AudioDecodeTab(), "Decode")
        lay = QVBoxLayout(self); lay.addWidget(tabs)
