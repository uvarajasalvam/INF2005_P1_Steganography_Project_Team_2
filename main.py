import os
import io
import zlib
import hmac
import math
import hashlib
import struct
import wave as wave_mod
from pathlib import Path

import numpy as np
from numpy.random import PCG64, Generator

from PIL import Image

from PySide6.QtCore import Qt, QRect, Signal
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtWidgets

from PySide6.QtGui import QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QGroupBox, QFileDialog, QLineEdit,
    QFormLayout, QTabWidget, QSplitter, QFrame, QMessageBox, QTextEdit,
    QRubberBand, QSlider, QSpinBox, QStackedWidget
)

# Matplotlib imports kept (optional; waveform plotting can be added later)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

# ----------------------- File types -----------------------
IMAGE_EXTS = {".bmp", ".png", ".gif", ".jpg", ".jpeg"}     # JPG/GIF preview only; embed later as PNG/BMP
AUDIO_EXTS = {".wav"}
COVER_EXTS = IMAGE_EXTS | AUDIO_EXTS

# ----------------------- Utils -----------------------

def human_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    for unit in ["KB", "MB", "GB", "TB"]:
        n /= 1024.0
        if n < 1024:
            return f"{n:.2f} {unit}"
    return f"{n:.2f} PB"

def sha256(b: bytes) -> bytes:
    return hashlib.sha256(b).digest()

def hkdf_sha256(key: bytes, salt: bytes, info: bytes, length: int) -> bytes:
    """HKDF (RFC 5869) using SHA-256."""
    prk = hmac.new(salt, key, hashlib.sha256).digest()
    okm = b""
    t = b""
    i = 1
    while len(okm) < length:
        t = hmac.new(prk, t + info + bytes([i]), hashlib.sha256).digest()
        okm += t
        i += 1
    return okm[:length]

def cover_fingerprint(path: str, n_bytes: int = 1_048_576) -> bytes:
    """SHA-256 over first N bytes of cover file (truncate to 16 bytes for header)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        remaining = n_bytes
        while remaining > 0:
            chunk = f.read(min(65536, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.digest()[:16]

def canonical_salt(lsb: int, roi_xywh: tuple[int,int,int,int] | None, cover_id: bytes, media_kind: str) -> bytes:
    """
    media_kind: 'image' or 'audio'
    For image: roi_xywh = (x0,y0,w,h)
    For audio: roi_xywh = (start, 0, length, 0)  (reuse tuple shape)
    """
    if roi_xywh is None:
        roi_xywh = (0, 0, 0, 0)
    x0, y0, w, h = roi_xywh
    s = f"kind:{media_kind}|roi:{x0},{y0},{w},{h}|lsb:{lsb}|cover_id".encode()
    return sha256(s + cover_id)

def kdf_from_key(user_key: str, salt: bytes) -> dict:
    """Derive key material and split: K_perm(16), K_bit(1), K_crypto(32), K_check(4), plus nonce(12)."""
    key_bytes = user_key.encode("utf-8")
    okm = hkdf_sha256(key_bytes, salt, b"stego-hkdf-v1", 16 + 1 + 32 + 4 + 12)
    off = 0
    K_perm = okm[off:off+16]; off += 16
    K_bit  = okm[off:off+1];  off += 1
    K_crypto = okm[off:off+32]; off += 32
    K_check  = okm[off:off+4]; off += 4
    nonce    = okm[off:off+12]; off += 12
    return {"K_perm": K_perm, "K_bit": K_bit, "K_crypto": K_crypto, "K_check": K_check, "nonce": nonce}

def rng_from_16_bytes(k16: bytes) -> Generator:
    """Turn 16 bytes into a deterministic uint128 for PCG64 seed."""
    seed_int = int.from_bytes(k16, "little", signed=False)
    return Generator(PCG64(seed_int))

def build_header(version: int, lsb: int, roi_xywh: tuple[int,int,int,int],
                 payload_len: int, cover_fp16: bytes, salt16: bytes,
                 nonce12: bytes, kcheck4: bytes) -> bytes:
    """
    Fixed-length header (little endian), 84 bytes:
    magic 'STG1'(4) + ver(1)+flags(1)+lsb(1)+pad(1) + roi(16) + payload_len(8)
    + cover_fp16(16) + salt16(16) + nonce12(12) + kcheck4(4) + crc32(4)
    """
    magic = b"STG1"
    flags = 0
    pad = 0
    x0, y0, w, h = roi_xywh
    header_wo_crc = (
        magic +
        struct.pack("<BBBB", version, flags, lsb, pad) +
        struct.pack("<IIII", x0, y0, w, h) +
        struct.pack("<Q", payload_len) +
        cover_fp16 +
        salt16 +
        nonce12 +
        kcheck4
    )
    crc = zlib.crc32(header_wo_crc) & 0xFFFFFFFF
    return header_wo_crc + struct.pack("<I", crc)

def parse_header(buf: bytes) -> dict:
    if len(buf) < 84 or buf[:4] != b"STG1":
        raise ValueError("Invalid or missing header.")
    version, flags, lsb, pad = struct.unpack("<BBBB", buf[4:8])
    x0, y0, w, h = struct.unpack("<IIII", buf[8:24])
    payload_len, = struct.unpack("<Q", buf[24:32])
    cover_fp16 = buf[32:48]
    salt16     = buf[48:64]
    nonce12    = buf[64:76]
    kcheck4    = buf[76:80]
    crc_stored, = struct.unpack("<I", buf[80:84])
    crc_calc = zlib.crc32(buf[:80]) & 0xFFFFFFFF
    if crc_calc != crc_stored:
        raise ValueError("Header CRC mismatch.")
    return {
        "version": version, "flags": flags, "lsb": lsb, "roi": (x0,y0,w,h),
        "payload_len": payload_len, "cover_fp16": cover_fp16,
        "salt16": salt16, "nonce12": nonce12, "kcheck4": kcheck4
    }

# ----------------------- Widgets -----------------------

class ImageView(QtWidgets.QFrame):
    """Image viewer with rubber-band ROI selection and click coordinates."""
    roiSelected = Signal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self._pix = QPixmap()
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._origin = None
        self._display_rect = QRect()  # where the pixmap is drawn within the widget
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)

    def setImage(self, pix: QPixmap | None):
        self._pix = pix if pix and not pix.isNull() else QPixmap()
        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), Qt.white)
        if self._pix.isNull():
            p.setPen(Qt.gray)
            p.drawText(self.rect(), Qt.AlignCenter, "Drop or load an image")
            return
        # keep aspect ratio
        w = self.width()
        h = self.height()
        pw = self._pix.width()
        ph = self._pix.height()
        scale = min(w / pw, h / ph)
        dw = int(pw * scale)
        dh = int(ph * scale)
        x0 = (w - dw) // 2
        y0 = (h - dh) // 2
        target = QtCore.QRect(x0, y0, dw, dh)
        self._display_rect = target
        p.drawPixmap(target, self._pix)

    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self._pix.isNull():
            return
        if e.button() == Qt.LeftButton:
            self._origin = e.position().toPoint()
            self._rubber.setGeometry(QRect(self._origin, QtCore.QSize()))
            self._rubber.show()

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self._rubber.isVisible() and self._origin is not None:
            rect = QRect(self._origin, e.position().toPoint()).normalized()
            self._rubber.setGeometry(rect)

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent):
        if not self._rubber.isVisible():
            return
        self._rubber.hide()
        if self._origin is None:
            return
        rect = self._rubber.geometry()
        self._origin = None
        if rect.width() <= 0 or rect.height() <= 0:
            return
        # map from widget coords -> image coords
        if self._display_rect.width() == 0 or self._display_rect.height() == 0:
            return
        pw, ph = self._pix.width(), self._pix.height()
        sx = pw / self._display_rect.width()
        sy = ph / self._display_rect.height()
        x0 = int((rect.left() - self._display_rect.left()) * sx)
        y0 = int((rect.top()  - self._display_rect.top())  * sy)
        w  = int(rect.width()  * sx)
        h  = int(rect.height() * sy)
        # clip to image bounds
        x0 = max(0, min(x0, pw - 1))
        y0 = max(0, min(y0, ph - 1))
        w = max(1, min(w, pw - x0))
        h = max(1, min(h, ph - y0))
        self.roiSelected.emit(x0, y0, w, h)

class DropLabel(QLabel):
    fileDropped = Signal(str)
    def __init__(self, title: str, exts: set[str] | None, parent=None):
        super().__init__(parent)
        self.exts = exts
        self.setAcceptDrops(True)
        self.setAlignment(Qt.AlignCenter)
        self.setText(f"Drop {title} here\n—or—\nClick to browse")
        self.setObjectName("DropLabel")
        self.setStyleSheet("""
            #DropLabel {
                border: 2px dashed #888; border-radius: 12px;
                padding: 16px; color: #444; font-size: 14px; background: #fafafa;
            }
            #DropLabel:hover { background: #f0f7ff; }
        """)

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

# ----------------------- Tabs -----------------------

class EncodeTab(QWidget):
    """
    Handles both images and audio as cover media.
    Payload can be entered as TEXT (textbox) or FILE (any type).
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cover_path: str | None = None
        self.cover_kind: str | None = None   # 'image' or 'audio'

        # Payload state
        self.payload_path: str | None = None
        self.payload_text: str = ""

        self.cover_qpix: QPixmap | None = None
        self.img_info = None          # {'size': (w,h), 'mode': 'RGBA', 'bands': 4}
        self.audio_info = None        # {'frames': int, 'channels': int, 'sampwidth': int}
        self.roi_img = None           # (x0,y0,w,h) for images

        # Left column
        left = QVBoxLayout()

        cov_box = QGroupBox("Cover Media (Image or WAV)")
        cov_v = QVBoxLayout()
        self.cover_drop = DropLabel("an image (.bmp/.png/.gif/.jpg) or audio (.wav)", COVER_EXTS)
        self.cover_drop.fileDropped.connect(self.load_cover)
        self.cover_info = QLabel("No cover loaded")
        self.cover_info.setWordWrap(True)
        cov_v.addWidget(self.cover_drop)
        cov_v.addWidget(self.cover_info)
        cov_box.setLayout(cov_v)

        # --- Payload box with tabs: Text | File ---
        pay_box = QGroupBox("Payload (Text or File)")
        pay_v = QVBoxLayout()
        self.payload_tabs = QTabWidget()
        # Text tab
        self.payload_text_edit = QTextEdit()
        self.payload_text_edit.setPlaceholderText("Type or paste text payload here…")
        self.payload_text_edit.textChanged.connect(self.on_payload_text_changed)
        self.payload_text_info = QLabel("Text bytes: 0")
        t_wrap = QWidget(); t_l = QVBoxLayout(t_wrap)
        t_l.addWidget(self.payload_text_edit)
        t_l.addWidget(self.payload_text_info)
        # File tab
        self.payload_drop = DropLabel("a payload file (any type)", None)  # accept any
        self.payload_drop.fileDropped.connect(self.load_payload_file)
        self.payload_file_info = QLabel("No payload file loaded")
        f_wrap = QWidget(); f_l = QVBoxLayout(f_wrap)
        f_l.addWidget(self.payload_drop)
        f_l.addWidget(self.payload_file_info)

        self.payload_tabs.addTab(t_wrap, "Text")
        self.payload_tabs.addTab(f_wrap, "File")
        self.payload_tabs.currentChanged.connect(self.update_capacity_label)

        pay_v.addWidget(self.payload_tabs)
        pay_box.setLayout(pay_v)

        ctrl = QGroupBox("Embedding Controls")
        form = QFormLayout()

        # LSB slider (1..8)
        self.lsb_slider = QSlider(Qt.Horizontal)
        self.lsb_slider.setRange(1, 8)
        self.lsb_slider.setValue(1)
        self.lsb_slider.setTickInterval(1)
        self.lsb_slider.setSingleStep(1)
        self.lsb_slider.setTickPosition(QSlider.TicksBelow)
        self.lsb_value = QLabel("1")
        lsb_row = QHBoxLayout(); lsb_row.addWidget(self.lsb_slider, 1); lsb_row.addWidget(self.lsb_value, 0)
        lsb_widget = QWidget(); lsb_widget.setLayout(lsb_row)
        self.lsb_slider.valueChanged.connect(lambda v: self.lsb_value.setText(str(v)))
        self.lsb_slider.valueChanged.connect(self.update_capacity_label)

        # Key input
        self.key_edit = QLineEdit(); self.key_edit.setPlaceholderText("Enter numeric/passphrase key (required)")

        # ROI area selector (stacked: image ROI label vs audio ROI spinboxes)
        self.roi_stack = QStackedWidget()
        # Image ROI view label
        self.roi_img_label = QLabel("Image ROI: not set (drag on image to select)")
        img_roi_wrap = QWidget(); img_l = QVBoxLayout(img_roi_wrap); img_l.addWidget(self.roi_img_label); img_l.addStretch(1)
        self.roi_stack.addWidget(img_roi_wrap)

        # Audio ROI controls
        audio_roi_wrap = QWidget()
        aform = QFormLayout(audio_roi_wrap)
        self.audio_start = QSpinBox(); self.audio_start.setRange(0, 0); self.audio_start.valueChanged.connect(self.update_audio_length_max)
        self.audio_len = QSpinBox(); self.audio_len.setRange(0, 0); self.audio_len.valueChanged.connect(self.update_capacity_label)
        aform.addRow("Audio start sample:", self.audio_start)
        aform.addRow("Audio length (samples):", self.audio_len)
        self.roi_stack.addWidget(audio_roi_wrap)

        form.addRow("Number of LSBs:", lsb_widget)
        form.addRow("Key:", self.key_edit)
        form.addRow("ROI:", self.roi_stack)
        ctrl.setLayout(form)

        cap_box = QGroupBox("Capacity (ROI-based)")
        cap_v = QVBoxLayout()
        self.cap_label = QLabel("Load cover + set ROI + add payload to calculate.")
        self.cap_label.setWordWrap(True)
        cap_v.addWidget(self.cap_label)
        cap_box.setLayout(cap_v)

        btns = QHBoxLayout()
        self.encode_btn = QPushButton("Encode → Stego")
        self.encode_btn.clicked.connect(self.on_encode)
        btns.addWidget(self.encode_btn)

        left.addWidget(cov_box)
        left.addWidget(pay_box)
        left.addWidget(ctrl)
        left.addWidget(cap_box)
        left.addLayout(btns)
        left.addStretch(1)

        # Right column (preview)
        right = QVBoxLayout()
        prev_box = QGroupBox("Preview (Image shows ROI by dragging; Audio shows basic info)")
        prev_v = QVBoxLayout()
        self.cover_view = ImageView()
        self.cover_view.roiSelected.connect(self.set_roi_from_image)
        self.preview_info = QLabel("")  # used for audio info
        self.preview_info.setWordWrap(True)

        # We'll show either the image view or the info label
        self.preview_stack = QStackedWidget()
        img_wrap = QWidget(); iw = QVBoxLayout(img_wrap); iw.addWidget(self.cover_view)
        info_wrap = QWidget(); iw2 = QVBoxLayout(info_wrap); iw2.addWidget(self.preview_info); iw2.addStretch(1)
        self.preview_stack.addWidget(img_wrap)   # index 0: image
        self.preview_stack.addWidget(info_wrap)  # index 1: audio/info

        prev_v.addWidget(self.preview_stack)
        prev_box.setLayout(prev_v)

        log_box = QGroupBox("Log")
        log_v = QVBoxLayout()
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        log_v.addWidget(self.log_edit)
        log_box.setLayout(log_v)

        right.addWidget(prev_box)
        right.addWidget(log_box)

        splitter = QSplitter()
        left_w = QWidget(); left_w.setLayout(left)
        right_w = QWidget(); right_w.setLayout(right)
        splitter.addWidget(left_w); splitter.addWidget(right_w)
        splitter.setSizes([440, 740])

        main = QVBoxLayout(self)
        main.addWidget(splitter)

    # ---------- Helpers for payload ----------
    def current_payload_mode(self) -> str:
        # 0 = Text, 1 = File
        return "text" if self.payload_tabs.currentIndex() == 0 else "file"

    def get_payload_bytes(self) -> bytes | None:
        mode = self.current_payload_mode()
        if mode == "text":
            data = self.payload_text_edit.toPlainText().encode("utf-8")
            return data if len(data) > 0 else None
        else:
            if not self.payload_path or not os.path.isfile(self.payload_path):
                return None
            with open(self.payload_path, "rb") as f:
                return f.read()

    # ---------- Actions ----------
    def on_payload_text_changed(self):
        self.payload_text = self.payload_text_edit.toPlainText()
        n = len(self.payload_text.encode("utf-8"))
        self.payload_text_info.setText(f"Text bytes: {n}")
        self.update_capacity_label()

    def load_payload_file(self, path: str):
        try:
            size = os.path.getsize(path)
            self.payload_path = path
            self.payload_file_info.setText(f"Path: {path}\nSize: {human_bytes(size)}")
            self.update_capacity_label()
            self.log(f"Loaded payload file: {path} ({human_bytes(size)})")
        except Exception as e:
            self.error(str(e))

    def load_cover(self, path: str):
        try:
            ext = Path(path).suffix.lower()
            if ext not in COVER_EXTS:
                raise ValueError("Unsupported media format.")
            self.cover_path = path
            self.roi_img = None
            self.audio_info = None
            self.img_info = None

            if ext in IMAGE_EXTS:
                self.cover_kind = "image"
                qpix = QPixmap(path)
                if qpix.isNull():
                    raise ValueError("Failed to load image.")
                self.cover_qpix = qpix
                self.cover_view.setImage(qpix)
                im = Image.open(path)
                self.img_info = {"size": im.size, "mode": im.mode, "bands": len(im.getbands())}
                self.cover_info.setText(
                    f"Path: {path}\nType: IMAGE | Format: {im.format}, Mode: {im.mode}, Size: {im.size[0]}x{im.size[1]}"
                )
                self.roi_img_label.setText("Image ROI: not set (drag on image to select)")
                self.preview_stack.setCurrentIndex(0)  # image
                self.roi_stack.setCurrentIndex(0)      # image ROI controls

            else:
                # WAV audio basic info via stdlib wave
                self.cover_kind = "audio"
                with wave_mod.open(path, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()   # bytes per sample
                    framerate = wf.getframerate()
                    n_frames  = wf.getnframes()
                self.audio_info = {"frames": n_frames, "channels": n_channels, "sampwidth": sampwidth, "rate": framerate}
                self.cover_info.setText(
                    f"Path: {path}\nType: AUDIO | WAV {n_channels}ch @ {framerate}Hz, {sampwidth*8}-bit, frames={n_frames}"
                )
                self.preview_info.setText(
                    "WAV loaded.\nSet ROI using start and length (samples).\n"
                    f"Total frames: {n_frames}, channels: {n_channels}, sample width: {sampwidth*8}-bit."
                )
                # Configure audio ROI spinboxes
                self.audio_start.blockSignals(True)
                self.audio_len.blockSignals(True)
                self.audio_start.setRange(0, max(0, n_frames-1))
                self.audio_start.setValue(0)
                self.audio_len.setRange(0, n_frames)
                self.audio_len.setValue(n_frames)
                self.audio_start.blockSignals(False)
                self.audio_len.blockSignals(False)
                self.preview_stack.setCurrentIndex(1)  # info
                self.roi_stack.setCurrentIndex(1)      # audio ROI controls

            self.cap_label.setText("Select ROI to evaluate capacity.")
            self.update_capacity_label()
            self.log(f"Loaded cover: {path}")
        except Exception as e:
            self.error(str(e))

    def update_audio_length_max(self, start_val: int):
        if self.audio_info:
            max_len = max(0, self.audio_info["frames"] - start_val)
            self.audio_len.setMaximum(max_len)
        self.update_capacity_label()

    def set_roi_from_image(self, x0: int, y0: int, w: int, h: int):
        if self.cover_kind != "image":
            return
        self.roi_img = (x0, y0, w, h)
        self.roi_img_label.setText(f"Image ROI: x={x0}, y={y0}, w={w}, h={h}")
        self.update_capacity_label()
        self.log(f"ROI selected (image): {self.roi_img}")

    def current_lsb(self) -> int:
        return self.lsb_slider.value()

    def _current_payload_bits(self) -> int | None:
        """Return payload size in bits for active mode, or None if not set."""
        if self.current_payload_mode() == "text":
            n = len(self.payload_text_edit.toPlainText().encode("utf-8"))
            return n * 8 if n > 0 else None
        else:
            if self.payload_path and os.path.isfile(self.payload_path):
                return os.path.getsize(self.payload_path) * 8
            return None

    def update_capacity_label(self):
        if not self.cover_path:
            self.cap_label.setText("Load a cover first.")
            return

        lsb = self.current_lsb()
        # Capacity
        if self.cover_kind == "image":
            if not self.img_info or not self.roi_img:
                self.cap_label.setText("For images: select ROI by dragging on the preview.")
                return
            x0, y0, w, h = self.roi_img
            channels = self.img_info["bands"]
            capacity_bits = w * h * channels * lsb
        elif self.cover_kind == "audio":
            if not self.audio_info:
                self.cap_label.setText("Load a WAV to compute capacity.")
                return
            start = self.audio_start.value()
            length = self.audio_len.value()
            length = max(0, min(length, self.audio_info["frames"] - start))
            channels = self.audio_info["channels"]
            capacity_bits = length * channels * lsb
        else:
            self.cap_label.setText("Unsupported cover kind.")
            return

        payload_bits = self._current_payload_bits()
        if payload_bits is None:
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\n"
                f"Add payload (Text or File) to check fit."
            )
        else:
            ok = payload_bits <= capacity_bits
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\n"
                f"Payload: {payload_bits} bits ({human_bytes(payload_bits//8)})\n"
                f"Result: {'OK' if ok else 'Too large'}"
            )
    def _capacity_bits_for(self, lsb: int, roi_xywh: tuple[int, int, int, int]) -> int:
        """Return available embedding capacity in bits for current cover and ROI."""
        if self.cover_kind == "image":
            _, _, w, h = roi_xywh
            channels = self.img_info["bands"]
            return w * h * channels * lsb
        elif self.cover_kind == "audio":
            # roi tuple stored as (start, 0, length, 0)
            length = roi_xywh[2]
            channels = self.audio_info["channels"]
            return length * channels * lsb
        return 0
    
    # ---------- Encode (stub) ----------
    def on_encode(self):
        if not self.cover_path:
            self.error("Please load a cover media.")
            return

        # Payload bytes from current mode
        payload_bytes = self.get_payload_bytes()
        if payload_bytes is None or len(payload_bytes) == 0:
            self.error("Please enter payload text or choose a payload file.")
            return

        key = self.key_edit.text().strip()
        if not key:
            self.error("Key is required.")
            return
        lsb = self.current_lsb()

        # Determine ROI + media kind
        if self.cover_kind == "image":
            if not self.roi_img:
                self.error("Please select an image ROI by dragging on the preview.")
                return
            roi_xywh = self.roi_img
        else:
            if not self.audio_info:
                self.error("Audio info not available.")
                return
            start = self.audio_start.value()
            length = self.audio_len.value()
            length = max(0, min(length, self.audio_info["frames"] - start))
            roi_xywh = (start, 0, length, 0)  # reuse tuple shape for salt/header

        HEADER_BYTES = 84  # matches build_header()
        header_bits = HEADER_BYTES * 8
        payload_bits = len(payload_bytes) * 8
        capacity_bits = self._capacity_bits_for(lsb, roi_xywh)
        if header_bits + payload_bits > capacity_bits:
            self.error(
                "Payload too large for the selected ROI and LSBs.\n\n"
                f"Capacity: {capacity_bits} bits\n"
                f"Header:   {header_bits} bits\n"
                f"Payload:  {payload_bits} bits\n"
                f"Needed:   {header_bits + payload_bits} bits"
            )
            return
        
        try:
            cover_id = cover_fingerprint(self.cover_path)
            full_salt = canonical_salt(lsb, roi_xywh, cover_id, self.cover_kind or "unknown")
            salt16 = full_salt[:16]
            kd = kdf_from_key(key, full_salt)
            K_perm, K_bit, K_crypto, K_check, nonce = kd["K_perm"], kd["K_bit"], kd["K_crypto"], kd["K_check"], kd["nonce"]

            payload_len = len(payload_bytes)

            header = build_header(
                version=1,
                lsb=lsb,
                roi_xywh=roi_xywh,
                payload_len=payload_len,
                cover_fp16=cover_id,
                salt16=salt16,
                nonce12=nonce,
                kcheck4=K_check
            )

            # TODO: For images: read pixels into numpy, permute with K_perm, embed header+payload bits into chosen LSBs
            #       For audio:  read PCM samples, permute index, embed bits into LSBs of samples
            # For now we only log the plan and show placeholders.

            if self.cover_kind == "image":
                self.cover_view.setImage(self.cover_qpix)  # no-op visual for now
            else:
                pass  # audio preview uses text

            src_desc = f"text:{payload_len}B" if self.current_payload_mode() == "text" else f"file:{self.payload_path} ({human_bytes(payload_len)})"
            self.log(f"Encode ready:\n"
                     f" kind={self.cover_kind}, lsb={lsb}, roi={roi_xywh}\n"
                     f" payload={src_desc}\n"
                     f" salt16={salt16.hex()} bit_rot={K_bit[0] % lsb}\n"
                     f" header_bytes={len(header)}")
            QMessageBox.information(self, "Encode (Placeholder)",
                                    "Header + key schedule derived.\nEmbedding not implemented yet in this stub.")

        except Exception as e:
            self.error(str(e))

    def log(self, msg: str):
        self.log_edit.append(msg)

    def error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.log(f"[ERROR] {msg}")

class DecodeTab(QWidget):
    """
    Accepts image/audio stego media + key. Decoding logic is stubbed but media handling is unified.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stego_path: str | None = None
        self.stego_kind: str | None = None

        root = QVBoxLayout(self)

        media_box = QGroupBox("Stego Media (Image or WAV)")
        mv = QVBoxLayout()
        self.stego_drop = DropLabel("a stego image (.bmp/.png/.gif/.jpg) or audio (.wav)", COVER_EXTS)
        self.stego_drop.fileDropped.connect(self.load_stego)
        self.media_info = QLabel("No stego file loaded")
        self.media_info.setWordWrap(True)
        mv.addWidget(self.stego_drop)
        mv.addWidget(self.media_info)
        media_box.setLayout(mv)

        ctrl_box = QGroupBox("Controls")
        form = QFormLayout()
        self.key_edit = QLineEdit(); self.key_edit.setPlaceholderText("Enter numeric/passphrase key")
        form.addRow("Key:", self.key_edit)
        ctrl_box.setLayout(form)

        btns = QHBoxLayout()
        self.inspect_btn = QPushButton("Inspect Header (Placeholder)")
        self.decode_btn  = QPushButton("Decode Payload (Placeholder)")
        self.inspect_btn.clicked.connect(self.on_inspect)
        self.decode_btn.clicked.connect(self.on_decode)
        btns.addWidget(self.inspect_btn)
        btns.addWidget(self.decode_btn)

        log_box = QGroupBox("Log")
        lv = QVBoxLayout()
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        lv.addWidget(self.log_edit)
        log_box.setLayout(lv)

        root.addWidget(media_box)
        root.addWidget(ctrl_box)
        root.addLayout(btns)
        root.addWidget(log_box)

    def load_stego(self, path: str):
        try:
            ext = Path(path).suffix.lower()
            if ext not in COVER_EXTS:
                raise ValueError("Unsupported stego media format.")
            self.stego_path = path
            self.stego_kind = "image" if ext in IMAGE_EXTS else "audio"

            if self.stego_kind == "image":
                im = Image.open(path)
                self.media_info.setText(
                    f"Path: {path}\nType: IMAGE | Format: {im.format}, Mode: {im.mode}, Size: {im.size[0]}x{im.size[1]}"
                )
            else:
                with wave_mod.open(path, "rb") as wf:
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    framerate = wf.getframerate()
                    n_frames  = wf.getnframes()
                self.media_info.setText(
                    f"Path: {path}\nType: AUDIO | WAV {n_channels}ch @ {framerate}Hz, {sampwidth*8}-bit, frames={n_frames}"
                )
            self.log(f"Loaded stego media: {path}")
        except Exception as e:
            self.error(str(e))

    def on_inspect(self):
        if not self.stego_path:
            self.error("Load a stego media first.")
            return
        QMessageBox.information(self, "Inspect (Placeholder)",
                                "Read the first bytes using your embedding scheme to parse the header.\n"
                                "Use parse_header() after reconstructing bit order with the key schedule.")

    def on_decode(self):
        if not self.stego_path:
            self.error("Load a stego media first.")
            return
        key = self.key_edit.text().strip()
        if not key:
            self.error("Key is required.")
            return
        QMessageBox.information(self, "Decode (Placeholder)",
                                "1) Re-derive salt from header + user key.\n"
                                "2) Rebuild permutation with K_perm.\n"
                                "3) Extract bits, validate K_check, decrypt if enabled, and save payload.")

    def log(self, msg: str):
        self.log_edit.append(msg)

    def error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.log(f"[ERROR] {msg}")

# ----------------------- Main Window -----------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stego Studio (Encode / Decode)")
        self.resize(1200, 760)

        self.tabs = QTabWidget()
        self.encode_tab = EncodeTab()
        self.decode_tab = DecodeTab()
        self.tabs.addTab(self.encode_tab, "Encode")
        self.tabs.addTab(self.decode_tab, "Decode")
        self.setCentralWidget(self.tabs)

        self._build_menu()
        self.statusBar().showMessage("Ready")

    def _build_menu(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        act_open_cover = QAction("Open Cover…", self)
        act_open_cover.triggered.connect(self._open_cover_dialog)
        file_menu.addAction(act_open_cover)

        act_open_stego = QAction("Open Stego…", self)
        act_open_stego.triggered.connect(self._open_stego_dialog)
        file_menu.addAction(act_open_stego)

        file_menu.addSeparator()
        act_quit = QAction("Exit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        tools = menubar.addMenu("&Tools")
        act_recap = QAction("Recalculate Capacity", self)
        act_recap.triggered.connect(self._recalc_capacity)
        tools.addAction(act_recap)

    def _open_cover_dialog(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Media (*.bmp *.png *.gif *.jpg *.jpeg *.wav)")
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                self.tabs.setCurrentWidget(self.encode_tab)
                self.encode_tab.load_cover(files[0])

    def _open_stego_dialog(self):
        dlg = QFileDialog(self)
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter("Media (*.bmp *.png *.gif *.jpg *.jpeg *.wav)")
        if dlg.exec():
            files = dlg.selectedFiles()
            if files:
                self.tabs.setCurrentWidget(self.decode_tab)
                self.decode_tab.load_stego(files[0])

    def _recalc_capacity(self):
        self.encode_tab.update_capacity_label()
        self.statusBar().showMessage("Capacity recalculated", 1500)

# ----------------------- Entry -----------------------

def main():
    import sys
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
