# audio_ui.py
# Audio Steganography GUI — Encode & Decode (PySide6)
# --------------------------------------------------
# What this does
# - WAV-only (PCM) steganography using LSB(s) over a selected ROI (start, length)
# - Keyed permutation and per-byte bit rotation derived via HKDF(passphrase, salt16)
# - Header-first layout (84 bytes) + raw payload bytes
# - Final Key token (stg1:...) contains public params so decoder can rebuild
#
# Notes
# - This implements a *self-contained* format (not compatible with steghide).
# - No payload encryption yet (K_crypto/nonce reserved for future AEAD).
# - cover_fp16 inside the header is informational here (binds salt during encode);
#   at decode we don't recompute/compare it because we don't have the pristine cover.
#
# Run
#   python audio_stego_gui.py

import os, math, zlib, hmac, hashlib, struct, base64, wave as wave_mod
from pathlib import Path

import numpy as np
from numpy.random import PCG64, Generator

from PySide6.QtCore import Qt, Signal
import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtWidgets
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QFileDialog, QLineEdit, QFormLayout, QTabWidget, QSplitter,
    QMessageBox, QTextEdit, QSlider, QSpinBox, QDoubleSpinBox
)

from pydub import AudioSegment
from pydub.utils import which
print(which("ffmpeg"))
import tempfile, uuid

# Matplotlib for waveform
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ---------- File types ----------
AUDIO_EXTS = {".wav"}
SUPPORTED_AUDIO_EXTS = AUDIO_EXTS.union({".mp3", ".flac", ".ogg"})


# Convert non-WAV audio files (mp3, flac, ogg) to temporary WAV
def convert_to_wav(filepath):
    # Get folder + base name of original file
    folder = os.path.dirname(filepath)
    base = os.path.splitext(os.path.basename(filepath))[0]

    # Output path (same folder, but .wav extension)
    wav_path = os.path.join(folder, base + ".wav")

    # Convert and save
    audio = AudioSegment.from_file(filepath)
    audio.export(wav_path, format="wav")

    return wav_path

# ---------- Utils (shared) ----------
def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"

def fmt_time(samples: int, rate: int) -> str:
    if rate <= 0:
        return "0:00"
    secs = max(0, samples) / rate
    m = int(secs // 60); s = int(round(secs % 60))
    if s == 60: m += 1; s = 0
    return f"{m}:{s:02d}"

def secs_to_frames(start_sec: float, dur_sec: float, rate: int, total_frames: int) -> tuple[int, int]:
    """Convert seconds to (start_frame, length_frames) using floor/ceil and clamp."""
    if rate <= 0 or total_frames <= 0:
        return 0, 1
    start = int(math.floor(max(0.0, start_sec) * rate))
    end   = int(math.ceil(max(start_sec + max(dur_sec, 0.0), 1e-9) * rate))
    start = min(start, max(0, total_frames - 1))
    if end <= start:
        end = start + 1
    end = min(end, total_frames)
    return start, end - start

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

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

# -------------------------------
# Header pack/unpack
# Header wo CRC = 80 bytes, with CRC = 84 bytes
HEADER_WO_CRC_LEN = 80
HEADER_TOTAL_LEN   = 84

HEADER_FMT_PREFIX  = b"STG1"

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
    if len(buf) != HEADER_TOTAL_LEN:
        raise ValueError("Header size mismatch")
    magic = buf[:4]
    if magic != HEADER_FMT_PREFIX:
        raise ValueError("Bad header magic")
    version, flags, lsb, pad = struct.unpack("<BBBB", buf[4:8])
    x0, y0, w, h = struct.unpack("<IIII", buf[8:24])
    payload_len, = struct.unpack("<Q", buf[24:32])
    cover_fp16 = buf[32:48]
    salt16     = buf[48:64]
    nonce12    = buf[64:76]
    kcheck4    = buf[76:80]
    crc32_read, = struct.unpack("<I", buf[80:84])

    calc = zlib.crc32(buf[:80]) & 0xFFFFFFFF
    if calc != crc32_read:
        raise ValueError("Header CRC32 mismatch (corrupted or wrong key/ROI/LSB)")

    return ParsedHeader(version, flags, lsb, (x0,y0,w,h), payload_len, cover_fp16, salt16, nonce12, kcheck4, crc32_read)

# -------------------------------
# Final Key token helpers

def make_key_token(media_kind: str, lsb: int, roi_xywh: tuple[int,int,int,int], salt16: bytes, kcheck4: bytes) -> str:
    media_code = 0 if media_kind == "image" else 1
    x0, y0, w, h = roi_xywh
    packed = struct.pack("<4sBBIIII16s4s", b"KEY1", media_code, lsb, x0, y0, w, h, salt16, kcheck4)
    b64 = base64.urlsafe_b64encode(packed).decode("ascii")
    return f"stg1:{b64}"

def parse_key_token(token: str) -> dict:
    if not token.startswith("stg1:"):
        raise ValueError("Invalid token prefix.")
    raw = base64.urlsafe_b64decode(token.split("stg1:",1)[1].encode("ascii"))
    magic, media_code, lsb, x0, y0, w, h, salt16, kcheck4 = struct.unpack("<4sBBIIII16s4s", raw)
    if magic != b"KEY1":
        raise ValueError("Bad token magic.")
    media_kind = "image" if media_code == 0 else "audio"
    return {"media_kind": media_kind, "lsb": lsb, "roi": (x0,y0,w,h), "salt16": salt16, "kcheck4": kcheck4}


# ---------- Widgets ----------
# Waveform widget (for ROI picking)
class WaveformView(QWidget):
    selectionChanged = Signal(int, int)  # (start_sample, length)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rate = 1
        self._x = None
        self._y = None
        self._span = None
        self._press_sample = None

        lay = QVBoxLayout(self)
        self.fig = Figure(figsize=(5, 2), facecolor="white")
        self.canvas = FigureCanvasQTAgg(self.fig)
        lay.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111)
        self._pretty_axes()
        self.fig.tight_layout()

        # Matplotlib events
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("button_release_event", self._on_release)

    def _pretty_axes(self):
        self.ax.set_facecolor("#f9fbff")
        self.ax.grid(True, alpha=0.25)
        for spine in ["top", "right"]:
            self.ax.spines[spine].set_visible(False)
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude")

    def set_audio(self, samples_mono: np.ndarray, rate: int):
        self._rate = int(rate)
        n = len(samples_mono)
        if n == 0:
            self.clear()
            return
        step = max(1, n // 200_000)
        y = samples_mono[::step].astype(np.float32)
        maxv = float(np.max(np.abs(y))) if y.size else 1.0
        if maxv <= 0: maxv = 1.0
        y = y / maxv
        x = (np.arange(len(y)) * step) / self._rate
        self._x, self._y = x, y
        self.ax.clear()
        self._pretty_axes()
        self.ax.plot(self._x, self._y, linewidth=1.25, color="r")
        self.ax.set_xlim(float(self._x[0]), float(self._x[-1]))
        self.ax.set_ylim(-1.1, 1.1)
        self._remove_span()
        self.canvas.draw_idle()

    def set_selection(self, start_sample: int, length: int):
        if self._rate <= 0 or self._x is None:
            return
        start_sample = max(0, start_sample)
        end_sample = max(start_sample + 1, start_sample + length)
        t0 = start_sample / self._rate
        t1 = end_sample / self._rate
        self._draw_span(t0, t1)

    # --- internal helpers ---
    def _on_press(self, ev):
        if ev.inaxes != self.ax or self._rate <= 0:
            return
        self._press_sample = int(max(0.0, ev.xdata) * self._rate)

    def _on_move(self, ev):
        if self._press_sample is None or ev.inaxes != self.ax:
            return
        cur_sample = int(max(0.0, ev.xdata) * self._rate)
        s0 = min(self._press_sample, cur_sample)
        s1 = max(self._press_sample, cur_sample)
        self._draw_span(s0 / self._rate, s1 / self._rate)

    def _on_release(self, ev):
        if self._press_sample is None or ev.inaxes != self.ax:
            self._press_sample = None
            return
        cur_sample = int(max(0.0, ev.xdata) * self._rate)
        s0 = max(0, min(self._press_sample, cur_sample))
        s1 = max(s0 + 1, max(self._press_sample, cur_sample))
        self._press_sample = None
        self.selectionChanged.emit(s0, s1 - s0)

    def _draw_span(self, t0: float, t1: float):
        self._remove_span()
        if t1 <= t0:
            return
        self._span = self.ax.axvspan(t0, t1, alpha=0.25, color="#1DB954")
        self.canvas.draw_idle()

    def _remove_span(self):
        if self._span is not None:
            try:
                self._span.remove()
            except Exception:
                pass
            self._span = None

    def clear(self):
        self.ax.clear()
        self._pretty_axes()
        self.canvas.draw_idle()
        self._x = self._y = None
        self._remove_span()

    def set_comparison(self, samples_orig: np.ndarray, samples_stego: np.ndarray):
        """Stacked plot: original waveform on top, stego below, properly spaced."""
        from matplotlib.gridspec import GridSpec

        self.fig.clf()  # Clear the figure

        # Create a GridSpec with 2 rows
        gs = GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)  # hspace adds space between plots

        n = min(len(samples_orig), len(samples_stego))
        orig_norm = samples_orig[:n] / np.max(np.abs(samples_orig[:n]))
        stego_norm = samples_stego[:n] / np.max(np.abs(samples_stego[:n]))
        t = np.arange(n) / self._rate

        # Original waveform (top)
        ax_orig = self.fig.add_subplot(gs[0])
        ax_orig.plot(t, orig_norm, color="b", linewidth=1)
        ax_orig.set_ylabel("Original")
        ax_orig.set_xlim(t[0], t[-1])
        ax_orig.set_ylim(-1.1, 1.1)
        ax_orig.grid(True, alpha=0.25)
        ax_orig.spines["top"].set_visible(False)
        ax_orig.spines["right"].set_visible(False)

        # Stego waveform (bottom)
        ax_stego = self.fig.add_subplot(gs[1], sharex=ax_orig)
        ax_stego.plot(t, stego_norm, color="r", linewidth=1)
        ax_stego.set_ylabel("Stego")
        ax_stego.set_xlabel("Time (s)")
        ax_stego.set_xlim(t[0], t[-1])
        ax_stego.set_ylim(-1.1, 1.1)
        ax_stego.grid(True, alpha=0.25)
        ax_stego.spines["top"].set_visible(False)
        ax_stego.spines["right"].set_visible(False)

        self.canvas.draw_idle()

# -------------------------------
# DropLabel (drag & drop picker)
class DropLabel(QLabel):
    fileDropped = Signal(str)
    def __init__(self, title: str, exts: set[str] | None, parent=None):
        super().__init__(parent)
        self.exts = exts
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"Drop {title} here\n—or—\nClick to browse")
        self.setObjectName("DropLabel")
        self.setStyleSheet(
            """
            #DropLabel { border: 2px dashed #888; border-radius: 12px; padding: 16px;
                         color: #444; font-size: 14px; background: #fafafa; }
            #DropLabel:hover { background: #f0f7ff; }
            """
        )

    def dragEnterEvent(self, e: QtGui.QDragEnterEvent):
        if e.mimeData().hasUrls():
            path = e.mimeData().urls()[0].toLocalFile()
            if self._acceptable(path):
                e.acceptProposedAction()

    def dropEvent(self, e: QtGui.QDropEvent):
        path = e.mimeData().urls()[0].toLocalFile()
        if self._acceptable(path):
            self.fileDropped.emit(path)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if e.button() == Qt.LeftButton:
            dlg = QFileDialog(self); dlg.setFileMode(QFileDialog.ExistingFile)
            if self.exts:
                patt = " ".join(f"*{x}" for x in sorted(self.exts))
                dlg.setNameFilter(f"Supported files ({patt})")
            if dlg.exec():
                files = dlg.selectedFiles()
                if files and self._acceptable(files[0]):
                    self.fileDropped.emit(files[0])

    def _acceptable(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False
        if self.exts is None:
            return True
        return Path(path).suffix.lower() in self.exts

# -------------------------------
# Payload panel (text/file)
class PayloadPanel(QWidget):
    def __init__(self, on_changed):
        super().__init__()
        self.payload_path: str | None = None
        box = QVBoxLayout(self)
        self.tabs = QTabWidget()
        # Text
        self.text_edit = QTextEdit();
        self.text_edit.setPlaceholderText("Type or paste text payload here…")
        self.text_info = QLabel("Text bytes: 0")
        self.text_edit.textChanged.connect(self._text_changed)
        t = QWidget(); tl = QVBoxLayout(t); tl.addWidget(self.text_edit); tl.addWidget(self.text_info)
        # File
        self.file_drop = DropLabel("a payload file (any type)", None)
        self.file_drop.fileDropped.connect(self._load_file)
        self.file_info = QLabel("No payload file loaded")
        f = QWidget(); fl = QVBoxLayout(f); fl.addWidget(self.file_drop); fl.addWidget(self.file_info)
        self.tabs.addTab(t, "Text"); self.tabs.addTab(f, "File")
        self.tabs.currentChanged.connect(on_changed)
        box.addWidget(self.tabs)
        self._on_changed = on_changed

    def mode(self) -> str:
        return "text" if self.tabs.currentIndex() == 0 else "file"

    def payload_bytes(self) -> bytes | None:
        if self.mode() == "text":
            b = self.text_edit.toPlainText().encode("utf-8")
            return b if b else None
        if self.payload_path and os.path.isfile(self.payload_path):
            return open(self.payload_path, "rb").read()
        return None

    def payload_bits(self) -> int | None:
        if self.mode() == "text":
            n = len(self.text_edit.toPlainText().encode("utf-8"))
            return n*8 if n>0 else None
        if self.payload_path and os.path.isfile(self.payload_path):
            return os.path.getsize(self.payload_path)*8
        return None

    def _text_changed(self):
        n = len(self.text_edit.toPlainText().encode("utf-8"))
        self.text_info.setText(f"Text bytes: {n}")
        self._on_changed()

    def _load_file(self, path: str):
        try:
            size = os.path.getsize(path)
            self.payload_path = path
            self.file_info.setText(f"Path: {path}\nSize: {human_bytes(size)}")
            self._on_changed()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))


# ---------- AUDIO ENCODE ----------
# Encode tab
class AudioEncodeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cover_path: str | None = None
        self.audio_info = None  # frames, channels, sampwidth, rate

        left = QVBoxLayout()
        cov_box = QGroupBox("Cover Audio (WAV, MP3, FLAC, OGG)")
        cv = QVBoxLayout()
        self.cover_drop = DropLabel("an audio file", SUPPORTED_AUDIO_EXTS)
        self.cover_drop.fileDropped.connect(self.load_cover)
        self.cover_info = QLabel("No audio loaded"); self.cover_info.setWordWrap(True)
        cv.addWidget(self.cover_drop); cv.addWidget(self.cover_info)
        cov_box.setLayout(cv)

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

        # Audio ROI controls (samples)
        self.audio_start = QSpinBox(); self.audio_start.setRange(0,0)
        self.audio_len   = QSpinBox(); self.audio_len.setRange(1,1)
        self.audio_start.valueChanged.connect(self._start_changed)
        self.audio_len.valueChanged.connect(self.update_capacity_label)

        # Friendly seconds controls
        self.start_sec = QDoubleSpinBox(); self.start_sec.setDecimals(3)
        self.start_sec.setRange(0.0, 0.0); self.start_sec.setSingleStep(0.010)
        self.start_sec.setSuffix(" s"); self.start_sec.valueChanged.connect(self._start_sec_changed)
        self.len_sec = QDoubleSpinBox(); self.len_sec.setDecimals(3)
        self.len_sec.setRange(0.001, 0.001); self.len_sec.setSingleStep(0.010)
        self.len_sec.setSuffix(" s"); self.len_sec.valueChanged.connect(self._len_sec_changed)

        form.addRow("Number of LSBs:", lsb_widget)
        form.addRow("Key:", self.key_edit)
        form.addRow("Start sample:", self.audio_start)
        form.addRow("Length (samples):", self.audio_len)
        form.addRow("Start (seconds):", self.start_sec)
        form.addRow("Length (seconds):", self.len_sec)
        ctrl.setLayout(form)

        # Timestamp scrubber
        ts_box = QGroupBox("Timestamp (slide to choose start)")
        tsv = QVBoxLayout()
        row = QHBoxLayout()
        self.lbl_time_left = QLabel("0:00")
        self.lbl_time_cur  = QLabel("0:00")
        self.lbl_time_right = QLabel("0:00")
        self.lbl_time_cur.setAlignment(Qt.AlignCenter)
        self.lbl_time_cur.setStyleSheet("font-weight:600;")
        row.addWidget(self.lbl_time_left, 0); row.addWidget(self.lbl_time_cur, 1); row.addWidget(self.lbl_time_right, 0)
        tsv.addLayout(row)
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setRange(0, 0); self.time_slider.setSingleStep(1); self.time_slider.setPageStep(44100)
        self.time_slider.setStyleSheet(
            """
            QSlider::groove:horizontal { height: 6px; background: #3a3a3a; border-radius: 3px; }
            QSlider::sub-page:horizontal { background: #1DB954; height: 6px; border-radius: 3px; }
            QSlider::add-page:horizontal { background: #3a3a3a; height: 6px; border-radius: 3px; }
            QSlider::handle:horizontal { background: #ffffff; border: 1px solid #1DB954; width: 14px; height: 14px; margin: -6px 0; border-radius: 7px; }
            """
        )
        self.time_slider.valueChanged.connect(self._time_slider_changed)
        tsv.addWidget(self.time_slider)
        ts_box.setLayout(tsv)

        cap_box = QGroupBox("Capacity")
        self.cap_label = QLabel("Load WAV + set ROI + add payload."); self.cap_label.setWordWrap(True)
        cap_v = QVBoxLayout(); cap_v.addWidget(self.cap_label); cap_box.setLayout(cap_v)

        key_box = QGroupBox("Final Key (copy for decoding)")
        key_h = QHBoxLayout()
        self.key_token_edit = QLineEdit(); self.key_token_edit.setReadOnly(True)
        self.copy_btn = QPushButton("Copy"); self.copy_btn.clicked.connect(self.copy_key_token)
        key_h.addWidget(self.key_token_edit); key_h.addWidget(self.copy_btn)
        key_box.setLayout(key_h)

        btns = QHBoxLayout()
        self.encode_btn = QPushButton("Encode")
        self.encode_btn.clicked.connect(self.on_encode)
        btns.addWidget(self.encode_btn)

        left.addWidget(cov_box); left.addWidget(pay_box); left.addWidget(ctrl)
        left.addWidget(ts_box); left.addWidget(cap_box); left.addWidget(key_box)
        left.addLayout(btns); left.addStretch(1)

        # RIGHT: waveform + log
        right = QVBoxLayout()
        wf_box = QGroupBox("Waveform (click–drag to select ROI)")
        wf_v = QVBoxLayout(); self.wave = WaveformView(); self.wave.selectionChanged.connect(self._on_wave_selection)
        wf_v.addWidget(self.wave); wf_box.setLayout(wf_v)

        log_box = QGroupBox("Log")
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        lv = QVBoxLayout(); lv.addWidget(self.log_edit); log_box.setLayout(lv)

        right.addWidget(wf_box); right.addWidget(log_box)

        splitter = QSplitter()
        lw = QWidget(); lw.setLayout(left)
        rw = QWidget(); rw.setLayout(right)
        splitter.addWidget(lw); splitter.addWidget(rw); splitter.setSizes([540, 680])
        main = QVBoxLayout(self); main.addWidget(splitter)

    def copy_key_token(self):
        QtGui.QGuiApplication.clipboard().setText(self.key_token_edit.text())
        QMessageBox.information(self, "Copied", "Final Key copied to clipboard.")

    # ----- WAV helpers
    def _read_wav_mono(self, path: str):
        with wave_mod.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames  = wf.getnframes()
            raw = wf.readframes(n_frames)
        if sampwidth == 1:
            a = np.frombuffer(raw, dtype=np.uint8).astype(np.int16); a = (a - 128) << 8
        elif sampwidth == 2:
            a = np.frombuffer(raw, dtype=np.int16)
        elif sampwidth == 3:
            b = np.frombuffer(raw, dtype=np.uint8)
            if len(b) % 3 != 0: b = b[: (len(b)//3)*3]
            b = b.reshape(-1, 3)
            a = (b[:,0].astype(np.int32) | (b[:,1].astype(np.int32) << 8) | (b[:,2].astype(np.int32) << 16))
            mask = 1 << 23; a = (a ^ mask) - mask
        elif sampwidth == 4:
            a = np.frombuffer(raw, dtype=np.int32)
        else:
            raise ValueError(f"Unsupported sample width: {sampwidth*8}-bit")
        if n_channels > 1:
            a = a.reshape(-1, n_channels).mean(axis=1).astype(a.dtype)
        info = {"frames": n_frames, "channels": n_channels, "sampwidth": sampwidth, "rate": framerate}
        return a, framerate, info

    def _read_wav_bytes(self, path: str):
        with wave_mod.open(path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth  = wf.getsampwidth()
            framerate  = wf.getframerate()
            n_frames   = wf.getnframes()
            raw = wf.readframes(n_frames)
        raw_u8 = np.frombuffer(raw, dtype=np.uint8).copy()
        params = {"channels": n_channels, "sampwidth": sampwidth, "rate": framerate, "frames": n_frames}
        return raw_u8, params

    def _read_audio_mono(self, path: str):
        ext = Path(path).suffix.lower()
        if ext in AUDIO_EXTS:  # existing WAV handling
            return self._read_wav_mono(path)
        else:  # MP3 / FLAC / OGG
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(audio.frame_rate)
            samples = np.array(audio.get_array_of_samples(), dtype=np.int16)
            info = {
                "frames": len(samples),
                "channels": 1,
                "sampwidth": 2,
                "rate": audio.frame_rate
            }
            return samples, audio.frame_rate, info

    def _read_audio_bytes(self, path: str):
        ext = Path(path).suffix.lower()
        if ext in AUDIO_EXTS:  # existing WAV handling
            return self._read_wav_bytes(path)
        else:
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(audio.channels).set_frame_rate(audio.frame_rate)
            raw = np.array(audio.get_array_of_samples())
            raw_u8 = raw.view(np.uint8).copy()
            params = {
                "channels": audio.channels,
                "sampwidth": 2,  # pydub default
                "rate": audio.frame_rate,
                "frames": len(raw)
            }
            return raw_u8, params

    def _write_wav_bytes(self, out_path: str, params: dict, raw_u8: np.ndarray):
        with wave_mod.open(out_path, "wb") as wf:
            wf.setnchannels(params["channels"]) ; wf.setsampwidth(params["sampwidth"]) ; wf.setframerate(params["rate"]) ; wf.writeframes(raw_u8.tobytes())

    # ----- Target indices (LSB byte per sample across channels)
    def _target_byte_indices(self, params: dict, start_frame: int, length_frames: int) -> np.ndarray:
        C = params["channels"]; B = params["sampwidth"]; F = length_frames
        bytes_per_frame = C * B
        base = start_frame * bytes_per_frame
        idx = np.empty(F * C, dtype=np.int64)
        k = 0
        for i in range(F):
            frame_base = base + i * bytes_per_frame
            for c in range(C):
                idx[k] = frame_base + c * B
                k += 1
        return idx

    # ----- Bit helpers
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
        mask = (1 << lsb) - 1
        mask_u8    = np.uint8(mask)
        invmask_u8 = np.uint8(~mask_u8)
        n_targets = len(target_idx)
        for i in range(n_targets):
            try:
                bits = next(bit_groups_iter)
            except StopIteration:
                break
            v = self._rotate_bits(bits, lsb, kbit_byte)
            j = target_idx[i]
            base = np.uint8(raw_u8[j]) & invmask_u8
            raw_u8[j] = np.uint8(base | np.uint8(v))

    # ----- GUI reactions
    def load_cover(self, path: str):
        try:
            ext = Path(path).suffix.lower()
            if ext not in SUPPORTED_AUDIO_EXTS:
                raise ValueError("Unsupported audio format.")
            # keep original path for fingerprinting (so token binds to original file)
            self.orig_cover_path = path

            # convert to WAV if needed (returns same path for WAV)
            wav_path = convert_to_wav(path)
            self.cover_path = wav_path

            samples_mono, rate, info = self._read_wav_mono(wav_path)
            self.audio_info = info
            self.cover_info.setText(
                f"Path: {path}\nWAV {info['channels']}ch @ {info['rate']}Hz, {info['sampwidth'] * 8}-bit, frames={info['frames']}"
            )

            # set UI controls based on the WAV we will embed into
            self.audio_start.blockSignals(True);
            self.audio_len.blockSignals(True)
            self.audio_start.setRange(0, max(0, info["frames"] - 1));
            self.audio_start.setValue(0)
            self.audio_len.setRange(1, info["frames"]);
            self.audio_len.setValue(info["frames"])
            self.audio_start.blockSignals(False);
            self.audio_len.blockSignals(False)

            total_dur = info["frames"] / info["rate"] if info["rate"] > 0 else 0.0
            min_dur = 1.0 / info["rate"] if info["rate"] > 0 else 0.001
            self.start_sec.blockSignals(True);
            self.len_sec.blockSignals(True)
            self.start_sec.setRange(0.0, max(0.0, total_dur))
            self.len_sec.setRange(min_dur, max(min_dur, total_dur))
            self.start_sec.setValue(0.0);
            self.len_sec.setValue(max(min_dur, total_dur))
            self.start_sec.blockSignals(False);
            self.len_sec.blockSignals(False)

            self.time_slider.blockSignals(True)
            self.time_slider.setRange(0, max(0, info["frames"] - 1))
            self.time_slider.setPageStep(max(1, info["rate"]))
            self.time_slider.setValue(0)
            self.time_slider.blockSignals(False)

            self.lbl_time_left.setText("0:00");
            self.lbl_time_right.setText(fmt_time(info["frames"], info["rate"]))
            self.lbl_time_cur.setText("0:00")
            self.wave.set_audio(samples_mono, rate)
            self.wave.set_selection(self.audio_start.value(), self.audio_len.value())
            self.update_capacity_label();
            self.log(f"Loaded audio: {path}")
        except Exception as e:
            self.error(str(e))

    def _on_wave_selection(self, start_sample: int, length: int):
        if self.audio_info:
            max_len = max(1, self.audio_info["frames"] - start_sample)
            length = max(1, min(length, max_len))
        self.audio_start.blockSignals(True); self.audio_len.blockSignals(True)
        self.audio_start.setValue(start_sample); self.audio_len.setValue(length)
        self.audio_start.blockSignals(False); self.audio_len.blockSignals(False)
        if self.audio_info:
            rate = self.audio_info["rate"]
            self.start_sec.blockSignals(True); self.len_sec.blockSignals(True)
            self.start_sec.setValue(start_sample / rate)
            self.len_sec.setValue(length / rate)
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
        rate = self.audio_info["rate"]
        self.start_sec.blockSignals(True); self.start_sec.setValue(val / rate); self.start_sec.blockSignals(False)
        self.wave.set_selection(self.audio_start.value(), self.audio_len.value()); self.update_capacity_label()

    def _start_changed(self, start_val: int):
        if self.audio_info:
            max_len = max(1, self.audio_info["frames"] - start_val)
            self.audio_len.setMaximum(max_len)
        self.wave.set_selection(self.audio_start.value(), self.audio_len.value())
        self._set_time_slider(self.audio_start.value())
        if self.audio_info:
            rate = self.audio_info["rate"]
            self.start_sec.blockSignals(True); self.len_sec.blockSignals(True)
            self.start_sec.setValue(self.audio_start.value() / rate)
            self.len_sec.setValue(self.audio_len.value() / rate)
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
        self._set_time_slider(start_f)
        self.wave.set_selection(start_f, len_f)
        max_len_sec = (total - start_f) / rate
        self.len_sec.blockSignals(True); self.len_sec.setMaximum(max_len_sec if max_len_sec > 0 else self.len_sec.minimum()); self.len_sec.blockSignals(False)
        self.update_capacity_label()

    def _len_sec_changed(self, d: float):
        if not self.audio_info: return
        rate = self.audio_info["rate"]; total = self.audio_info["frames"]
        start_f, len_f = secs_to_frames(self.start_sec.value(), d, rate, total)
        self.audio_len.blockSignals(True); self.audio_len.setValue(len_f); self.audio_len.blockSignals(False)
        self.wave.set_selection(start_f, len_f); self.update_capacity_label()

    def current_lsb(self) -> int:
        return self.lsb_slider.value()

    def update_capacity_label(self):
        if not self.cover_path or not self.audio_info:
            self.cap_label.setText("Load WAV + set ROI + add payload."); return
        lsb = self.current_lsb(); start = self.audio_start.value(); length = self.audio_len.value()
        length = max(1, min(length, self.audio_info["frames"] - start))
        channels = self.audio_info["channels"]
        capacity_bits = length * channels * lsb
        payload_bits = self.payload_panel.payload_bits()
        if payload_bits is None:
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\nStart {self.start_sec.value():.3f}s, Length {self.len_sec.value():.3f}s\nAdd payload (Text or File)."
            )
        else:
            ok = payload_bits <= capacity_bits
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\nStart {self.start_sec.value():.3f}s, Length {self.len_sec.value():.3f}s\nPayload: {payload_bits} bits ({human_bytes(payload_bits//8)})\nResult: {'OK' if ok else 'Too large'}"
            )

    def on_encode(self):
        if not self.cover_path or not self.audio_info:
            self.error("Load a cover audio file.")
            return
        payload = self.payload_panel.payload_bytes()
        if not payload:
            self.error("Enter payload text or choose a payload file.")
            return
        key = self.key_edit.text().strip()
        if not key:
            self.error("Key is required.")
            return

        try:
            lsb = self.current_lsb()
            start = self.audio_start.value()
            length = self.audio_len.value()
            length = max(1, min(length, self.audio_info["frames"] - start))
            roi = (start, 0, length, 0)
            cover_id = cover_fingerprint(getattr(self, "orig_cover_path", self.cover_path))
            full_salt = canonical_salt(lsb, roi, cover_id, "audio")
            salt16 = full_salt[:16]
            kd = kdf_from_key(key, salt16)
            K_perm, K_bit, K_crypto, K_check, nonce = kd["K_perm"], kd["K_bit"], kd["K_crypto"], kd["K_check"], kd[
                "nonce"]
            header = build_header(1, lsb, roi, len(payload), cover_id, salt16, nonce, K_check)
            token = make_key_token("audio", lsb, roi, salt16, K_check)
            self.key_token_edit.setText(token)

            # Read original WAV
            raw_u8, params = self._read_wav_bytes(self.cover_path)
            tgt = self._target_byte_indices(params, start, length)
            total_bits = (len(header) + len(payload)) * 8
            capacity_bits = len(tgt) * lsb
            if total_bits > capacity_bits:
                self.error(f"Not enough capacity in ROI.\nNeed {total_bits} bits, have {capacity_bits} bits.")
                return

            rng_seed = int.from_bytes(K_perm, "little", signed=False)
            rng = Generator(PCG64(rng_seed))
            perm = rng.permutation(len(tgt))
            tgt_perm = tgt[perm]
            bit_groups = self._bit_chunks(header + payload, lsb)
            kbit_byte = K_bit[0]
            self._embed_bits_into_bytes(raw_u8, tgt_perm, bit_groups, lsb, kbit_byte)

            stem = Path(self.cover_path).with_suffix("")
            out_path = str(stem) + "_stego.wav"
            self._write_wav_bytes(out_path, params, raw_u8)

            # --- Display waveform and overlay comparison ---
            orig_samples, rate, _ = self._read_wav_mono(self.cover_path)
            stego_samples, _, _ = self._read_wav_mono(out_path)

            # Show stego waveform
            self.wave.set_audio(stego_samples, rate)
            self.wave.set_selection(self.audio_start.value(), self.audio_len.value())

            # Overlay original and stego
            if hasattr(self.wave, "set_comparison"):
                self.wave.set_comparison(orig_samples, stego_samples)

            # Optional numeric comparison
            diff = np.abs(orig_samples.astype(np.int32) - stego_samples.astype(np.int32))
            self.log(f"Waveform comparison: max diff={diff.max()}, mean diff={diff.mean():.2f}")

            self.log(f"Derived token: {token}")
            self.log(
                f"Header bytes={len(header)} salt16={salt16.hex()} bit_rot={K_bit[0] % lsb} order_seed={int.from_bytes(K_perm, 'little')}")
            self.log(f"Embedded {total_bits} bits into {len(tgt)} target bytes @ {lsb} LSB(s).")
            self.log(f"Saved stego audio: {out_path}")
            QMessageBox.information(self, "Encode complete",
                                    f"Stego written:\n{out_path}\n\nCopy the Final Key for decoding.")
        except Exception as e:
            self.error(str(e))

    def log(self, msg: str):
        self.log_edit.append(msg)

    def error(self, msg: str):
        QMessageBox.critical(self, "Error", msg); self.log(f"[ERROR] {msg}")

# -------------------------------
# Decode tab
class AudioDecodeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.stego_path: str | None = None
        root = QVBoxLayout(self)
        media_box = QGroupBox("Stego Audio (WAV)")
        mv = QVBoxLayout()
        self.stego_drop = DropLabel("a stego audio file", SUPPORTED_AUDIO_EXTS)
        self.stego_drop.fileDropped.connect(self.load_stego)
        self.media_info = QLabel("No stego WAV loaded"); self.media_info.setWordWrap(True)
        mv.addWidget(self.stego_drop); mv.addWidget(self.media_info); media_box.setLayout(mv)

        ctrl = QGroupBox("Controls")
        form = QFormLayout()
        self.key_token_edit = QLineEdit(); self.key_token_edit.setPlaceholderText("Paste Final Key token (stg1:...)")
        self.user_key_edit = QLineEdit(); self.user_key_edit.setPlaceholderText("Enter the same numeric/passphrase key")
        form.addRow("Final Key:", self.key_token_edit)
        form.addRow("User Key:", self.user_key_edit)
        ctrl.setLayout(form)

        btns = QHBoxLayout()
        self.inspect_btn = QPushButton("Inspect Header")
        self.decode_btn  = QPushButton("Decode Payload")
        self.inspect_btn.clicked.connect(self.on_inspect)
        self.decode_btn.clicked.connect(self.on_decode)
        btns.addWidget(self.inspect_btn); btns.addWidget(self.decode_btn)

        log_box = QGroupBox("Log")
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        lv = QVBoxLayout(); lv.addWidget(self.log_edit); log_box.setLayout(lv)

        root.addWidget(media_box); root.addWidget(ctrl); root.addLayout(btns); root.addWidget(log_box)

    # ---- WAV helpers (decode side)
    def _read_wav_bytes(self, path: str):
        with wave_mod.open(path, "rb") as wf:
            n_channels = wf.getnchannels(); sampwidth = wf.getsampwidth(); framerate = wf.getframerate(); n_frames = wf.getnframes(); raw = wf.readframes(n_frames)
        raw_u8 = np.frombuffer(raw, dtype=np.uint8).copy()
        params = {"channels": n_channels, "sampwidth": sampwidth, "rate": framerate, "frames": n_frames}
        return raw_u8, params

    def _target_byte_indices(self, params: dict, start_frame: int, length_frames: int) -> np.ndarray:
        C = params["channels"]; B = params["sampwidth"]; F = length_frames
        bytes_per_frame = C * B
        base = start_frame * bytes_per_frame
        idx = np.empty(F * C, dtype=np.int64)
        k = 0
        for i in range(F):
            frame_base = base + i * bytes_per_frame
            for c in range(C):
                idx[k] = frame_base + c * B; k += 1
        return idx

    # ---- Extraction helpers
    @staticmethod
    def _bits_from_byte(value_u8: int, lsb: int) -> list[int]:
        # returns lsb bits MSB-first inside the field
        field = value_u8 & ((1 << lsb) - 1)
        out = []
        for i in range(lsb-1, -1, -1):
            out.append((field >> i) & 1)
        return out

    @staticmethod
    def _rotate_right(bits: list[int], lsb: int, off: int) -> list[int]:
        if lsb <= 0: return bits
        r = off % lsb
        if r == 0: return bits
        # left-rotate by k during encode -> right-rotate by k during decode
        return bits[-r:] + bits[:-r]

    @staticmethod
    def _bits_to_bytes(msb_bits: list[int]) -> bytes:
        # consume groups of 8 (pad final with zeros)
        out = bytearray()
        i = 0
        n = len(msb_bits)
        while i < n:
            val = 0
            for j in range(8):
                val = (val << 1) | (msb_bits[i+j] if (i+j) < n else 0)
            out.append(val)
            i += 8
        return bytes(out)

    def load_stego(self, path: str):
        try:
            ext = Path(path).suffix.lower()
            if ext not in AUDIO_EXTS: raise ValueError("Unsupported audio format.")
            with wave_mod.open(path, "rb") as wf:
                n_channels = wf.getnchannels(); sampwidth = wf.getsampwidth(); framerate = wf.getframerate(); n_frames  = wf.getnframes()
            self.stego_path = path
            self.media_info.setText(f"Path: {path}\nWAV {n_channels}ch @ {framerate}Hz, {sampwidth*8}-bit, frames={n_frames}")
            self.log(f"Loaded stego WAV: {path}")
        except Exception as e:
            self.error(str(e))

    def _extract_bits(self, raw_u8, tgt_indices, total_bits_needed, lsb, kbit_byte):
        """Extracts total_bits_needed bits in the *same permuted order* used by encode."""
        bits_out = []
        rotate = kbit_byte % lsb if lsb > 0 else 0
        for j in tgt_indices:
            field_bits = self._bits_from_byte(int(raw_u8[j]), lsb)
            # undo the left-rotation from encode
            if rotate:
                field_bits = self._rotate_right(field_bits, lsb, rotate)
            bits_out.extend(field_bits)
            if len(bits_out) >= total_bits_needed:
                return bits_out[:total_bits_needed]
        return bits_out

    def _rebuild_indices_perm(self, params, roi, K_perm):
        x0, _, w, _ = roi
        tgt = self._target_byte_indices(params, x0, w)
        rng_seed = int.from_bytes(K_perm, "little", signed=False)
        rng = Generator(PCG64(rng_seed))
        perm = rng.permutation(len(tgt))
        return tgt[perm]

    def _derive_keys_from_token(self, token: str, user_key: str):
        info = parse_key_token(token)
        kd = kdf_from_key(user_key, info["salt16"])  # HKDF(pass, salt16)
        # NEW: compare token's kcheck4 with derived K_check before any extraction
        if info["kcheck4"] != kd["K_check"]:
            raise ValueError("Wrong passphrase or tampered Final Key (K_check mismatch)")
        return info, kd

    def on_inspect(self):
        if not self.stego_path:
            self.error("Load a stego WAV first."); return
        token = self.key_token_edit.text().strip(); user_key = self.user_key_edit.text().strip()
        if not token or not user_key:
            self.error("Provide both Final Key token and the user key."); return
        try:
            info, kd = self._derive_keys_from_token(token, user_key)
            raw_u8, params = self._read_wav_bytes(self.stego_path)
            lsb = info["lsb"]; roi = info["roi"]; K_perm = kd["K_perm"]; K_bit = kd["K_bit"]; K_check = kd["K_check"]
            # capacity sanity for header
            tgt_perm = self._rebuild_indices_perm(params, roi, K_perm)
            capacity_bits = len(tgt_perm) * lsb
            if capacity_bits < HEADER_TOTAL_LEN*8:
                raise ValueError("ROI too small to contain header")
            # Extract header
            header_bits = self._extract_bits(raw_u8, tgt_perm, HEADER_TOTAL_LEN*8, lsb, K_bit[0])
            header_bytes = self._bits_to_bytes(header_bits)
            ph = parse_header(header_bytes)
            # quick key check
            if ph.kcheck4 != K_check:
                raise ValueError("Wrong key/token (K_check mismatch)")
            # consistency with token
            ok_ctx = (ph.lsb == lsb and ph.roi == roi)
            self.log(f"Header OK: version={ph.version} payload_len={ph.payload_len} lsb={ph.lsb} roi={ph.roi} salt16={ph.salt16.hex()} crc32=0x{ph.crc32:08x}")
            self.log("Context matches token: " + ("YES" if ok_ctx else "NO (mismatch)"))
            QMessageBox.information(self, "Inspect", "Header parsed successfully. See log for details.")
        except Exception as e:
            self.error(str(e))

    def convert_to_wav(path_in: str) -> str:
        """If input is already WAV, return it. Otherwise convert to a temp WAV and return path."""
        ext = Path(path_in).suffix.lower()
        if ext == ".wav":
            return path_in
        audio = AudioSegment.from_file(path_in)  # requires ffmpeg available
        tmp = Path(tempfile.gettempdir()) / f"{Path(path_in).stem}_{uuid.uuid4().hex}.wav"
        audio.export(str(tmp), format="wav")
        return str(tmp)

    def on_decode(self):
        if not self.stego_path:
            self.error("Load a stego WAV first."); return
        token = self.key_token_edit.text().strip(); user_key = self.user_key_edit.text().strip()
        if not token or not user_key:
            self.error("Provide both Final Key token and the user key."); return
        try:
            info, kd = self._derive_keys_from_token(token, user_key)
            if info["media_kind"] != "audio":
                raise ValueError("This token is not for audio")
            raw_u8, params = self._read_wav_bytes(self.stego_path)
            lsb = info["lsb"]; roi = info["roi"]; K_perm = kd["K_perm"]; K_bit = kd["K_bit"]; K_check = kd["K_check"]
            tgt_perm = self._rebuild_indices_perm(params, roi, K_perm)
            capacity_bits = len(tgt_perm) * lsb
            if capacity_bits < HEADER_TOTAL_LEN*8:
                raise ValueError("ROI too small to contain header")
            # 1) Extract header
            header_bits = self._extract_bits(raw_u8, tgt_perm, HEADER_TOTAL_LEN*8, lsb, K_bit[0])
            header_bytes = self._bits_to_bytes(header_bits)
            ph = parse_header(header_bytes)
            if ph.kcheck4 != K_check:
                raise ValueError("Wrong key/token (K_check mismatch)")
            if ph.lsb != lsb or ph.roi != roi:
                raise ValueError("Token/controls do not match embedded header")
            # 2) Extract payload
            need_bits = ph.payload_len * 8
            total_needed = HEADER_TOTAL_LEN*8 + need_bits
            if capacity_bits < total_needed:
                raise ValueError("ROI capacity too small for embedded payload")
            # We already consumed header_bits from the start of tgt_perm. Continue from there:
            # For simplicity, re-extract total_needed and slice (cheap and deterministic)
            all_bits = self._extract_bits(raw_u8, tgt_perm, total_needed, lsb, K_bit[0])
            payload_bits = all_bits[HEADER_TOTAL_LEN*8: HEADER_TOTAL_LEN*8 + need_bits]
            payload_bytes = self._bits_to_bytes(payload_bits)
            # Try UTF-8 display; otherwise offer to save
            try:
                text = payload_bytes.decode("utf-8")
                self.log(f"Decoded text payload ({len(payload_bytes)} bytes):\n{text}")
                QMessageBox.information(self, "Decode", f"Extracted text payload ({len(payload_bytes)} bytes). See Log.")
            except UnicodeDecodeError:
                # ask where to save
                out_path, _ = QFileDialog.getSaveFileName(self, "Save payload as…", "payload.bin")
                if out_path:
                    with open(out_path, "wb") as f: f.write(payload_bytes)
                    self.log(f"Saved binary payload to: {out_path} ({len(payload_bytes)} bytes)")
                    QMessageBox.information(self, "Decode", f"Binary payload saved ({len(payload_bytes)} bytes).")
                else:
                    self.log("[INFO] Binary payload not saved (user canceled).")
        except Exception as e:
            self.error(str(e))

    def log(self, msg: str):
        self.log_edit.append(msg)

    def error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.log(f"[ERROR] {msg}")

# -------------------------------
# Suite
class AudioSuite(QWidget):
    def __init__(self):
        super().__init__()
        tabs = QTabWidget(); tabs.addTab(AudioEncodeTab(), "Encode"); tabs.addTab(AudioDecodeTab(), "Decode")
        lay = QVBoxLayout(self); lay.addWidget(tabs)

