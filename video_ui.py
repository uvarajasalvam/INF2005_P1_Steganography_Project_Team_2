#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video Steganography GUI — Final
- ROI is drawn directly on the video; left sidebar shows live ROI thumbnail.
- Encode: LSB slider (1–4), key, start/length/frame-step, timestamp scrubber, capacity.
- Encode outputs a **Final Key token**; copy it to decode.
- Decode: paste Final Key token + enter the same key → recover payload.
- If payload is text (UTF-8), it prints into the decode log and saves a .txt, else a .bin.
- Optional side-by-side compare (cover vs stego) after decoding.

Requirements:
  pip install PySide6 opencv-python numpy
"""

import os, math, zlib, hmac, hashlib, struct, base64, time
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import cv2
import tempfile, subprocess, shutil

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QGroupBox, QLineEdit, QFormLayout, QTabWidget, QSplitter,
    QMessageBox, QTextEdit, QSlider, QSpinBox, QDoubleSpinBox, QFileDialog,
    QRubberBand, QComboBox, QCheckBox, QDialog
)

# -----------------------------------------------------------------------------#
APP_ROOT = Path.cwd()
OUTPUT_DIR = APP_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SUPPORTED_VIDEO_EXTS = {".mp4",".mkv",".avi",".mov",".m4v"}

def human_bytes(n: int) -> str:
    if n < 1024: return f"{n} B"
    for u in ["KB","MB","GB","TB"]:
        n /= 1024.0
        if n < 1024: return f"{n:.2f} {u}"
    return f"{n:.2f} PB"

def fmt_time(frames: int, fps: float) -> str:
    if fps <= 0: return "0:00"
    secs = max(0, frames) / fps
    m = int(secs // 60); s = int(round(secs % 60))
    if s == 60: m += 1; s = 0
    return f"{m}:{s:02d}"

def secs_to_frames(start_sec: float, dur_sec: float, fps: float, total_frames: int) -> tuple[int,int]:
    if fps <= 0 or total_frames <= 0: return 0, 1
    start = int(math.floor(max(0.0, start_sec) * fps))
    end   = int(math.ceil(max(start_sec + max(dur_sec,0.0), 1e-9) * fps))
    start = min(start, max(0, total_frames-1))
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

# -----------------------------------------------------------------------------#
# KDF / token with ROI + frame window + step + lsb + salts
def kdf_from_key(user_key: str, salt: bytes) -> dict:
    okm = hkdf_sha256(user_key.encode("utf-8"), salt, b"stego-hkdf-v1", 16 + 1 + 32 + 4 + 12)
    off = 0
    K_perm  = okm[off:off+16]; off += 16
    K_bit   = okm[off:off+1];  off += 1
    K_crypt = okm[off:off+32]; off += 32  # reserved
    K_check = okm[off:off+4];  off += 4
    nonce   = okm[off:off+12]; off += 12
    return {"K_perm":K_perm, "K_bit":K_bit, "K_check":K_check, "nonce":nonce}

def canonical_salt_video(lsb: int, roi_xywh: tuple[int,int,int,int],
                         frames_xyz: tuple[int,int,int], cover_id: bytes) -> bytes:
    x0,y0,w,h = roi_xywh
    f0,flen,fstep = frames_xyz
    s = f"kind:video|roi:{x0},{y0},{w},{h}|frames:{f0},{flen}|step:{fstep}|lsb:{lsb}|cover_id".encode()
    return sha256(s + cover_id)

# token format (KEY3: with step)
KEY3_FMT = "<4sBBIIIIIII16s4s"   # 'KEY3', media=1, lsb, x0,y0,w,h,start,len,step,salt16,kcheck4
def make_key_token_video(lsb: int, roi: tuple[int,int,int,int], start_f: int, len_f: int, step: int,
                         salt16: bytes, kcheck4: bytes) -> str:
    x0,y0,w,h = roi
    packed = struct.pack(KEY3_FMT, b"KEY3", 1, lsb, x0,y0,w,h, start_f, len_f, step, salt16, kcheck4)
    b64 = base64.urlsafe_b64encode(packed).decode("ascii")
    return f"stg1:{b64}"

def parse_key_token(token: str) -> dict:
    if not token.startswith("stg1:"): raise ValueError("Bad token prefix")
    raw = base64.urlsafe_b64decode(token.split("stg1:",1)[1].encode("ascii"))
    magic, media_code, lsb, x0,y0,w,h, start_f, len_f, step, salt16, kcheck4 = struct.unpack(KEY3_FMT, raw)
    if magic != b"KEY3": raise ValueError("Unsupported token")
    return {"media_kind":"video" if media_code==1 else "image",
            "lsb":lsb, "roi":(x0,y0,w,h), "start_f":start_f, "len_f":len_f, "step":step,
            "salt16":salt16, "kcheck4":kcheck4}

# -----------------------------------------------------------------------------#
# Header (84 bytes — same layout as earlier)
HEADER_FMT_PREFIX  = b"STG1"
HEADER_TOTAL_LEN   = 84
def build_header(version: int, lsb: int, roi_xywh: tuple[int,int,int,int],
                 payload_len: int, cover_fp16: bytes, salt16: bytes,
                 nonce12: bytes, kcheck4: bytes) -> bytes:
    x0,y0,w,h = roi_xywh
    header_wo = (
        HEADER_FMT_PREFIX +
        struct.pack("<BBBB", version, 0, lsb, 0) +
        struct.pack("<IIII", x0, y0, w, h) +
        struct.pack("<Q", payload_len) +
        cover_fp16 + salt16 + nonce12 + kcheck4
    )
    crc = zlib.crc32(header_wo) & 0xFFFFFFFF
    return header_wo + struct.pack("<I", crc)

class ParsedHeader:
    __slots__=("version","flags","lsb","roi","payload_len","cover_fp16","salt16","nonce12","kcheck4","crc32")
    def __init__(self, version, flags, lsb, roi, payload_len, cover_fp16, salt16, nonce12, kcheck4, crc32):
        self.version=version; self.flags=flags; self.lsb=lsb; self.roi=roi; self.payload_len=payload_len
        self.cover_fp16=cover_fp16; self.salt16=salt16; self.nonce12=nonce12; self.kcheck4=kcheck4; self.crc32=crc32

def parse_header(buf: bytes, strict_crc: bool=True) -> ParsedHeader:
    if len(buf) != HEADER_TOTAL_LEN: raise ValueError("Header length mismatch")
    if buf[:4] != HEADER_FMT_PREFIX: raise ValueError("Bad header magic")
    version, flags, lsb, pad = struct.unpack("<BBBB", buf[4:8])
    x0,y0,w,h = struct.unpack("<IIII", buf[8:24])
    payload_len, = struct.unpack("<Q", buf[24:32])
    cover_fp16 = buf[32:48]; salt16=buf[48:64]; nonce12=buf[64:76]; kcheck4=buf[76:80]
    crc_read, = struct.unpack("<I", buf[80:84])
    if strict_crc:
        calc = zlib.crc32(buf[:80]) & 0xFFFFFFFF
        if calc != crc_read: raise ValueError("Header CRC mismatch")
    return ParsedHeader(version, flags, lsb, (x0,y0,w,h), payload_len, cover_fp16, salt16, nonce12, kcheck4, crc_read)

# --- Safe capacity helpers (header reservation aware) ---
def header_fields_needed(lsb: int) -> int:
    """How many LSB-fields (in one channel) the header needs."""
    return (HEADER_TOTAL_LEN * 8 + lsb - 1) // lsb  # ceil(bits/lsb)

def safe_payload_capacity_bits(lsb: int, w: int, h: int, n_frames: int) -> int:
    """
    Total available payload bits after reserving the header area on the first frame's GREEN channel.
    Each pixel-channel is 1 field of size 'lsb' bits.
    """
    total_fields = w * h * 3 * n_frames              # all channels, all frames
    reserved = header_fields_needed(lsb)             # fields taken by header on GREEN of frame 0
    # Header must fit inside the ROI's GREEN channel of a single frame:
    if reserved > (w * h):
        return 0  # ROI too small for header
    usable_fields = total_fields - reserved
    return max(0, usable_fields * lsb)

# ---------- Payload envelope helpers ----------
ENV_MAGIC = b"ENV1"   # envelope format v1

def _sanitize_filename(name: str) -> str:
    # keep basename only, strip odd chars
    safe = Path(name).name
    safe = "".join(c for c in safe if c.isalnum() or c in ("-", "_", ".", " ", "(", ")"))
    return safe or "payload.bin"

def make_envelope(is_text: bool, data: bytes, filename: str | None) -> bytes:
    """
    ENV1 | typ(1B) | name_len(2B little) | name_bytes | data...
      typ: 1=text, 2=file
    """
    typ = 1 if is_text else 2
    name_bytes = (filename or "").encode("utf-8")
    if len(name_bytes) > 65535:
        name_bytes = name_bytes[:65535]
    hdr = ENV_MAGIC + bytes([typ]) + struct.pack("<H", len(name_bytes))
    return hdr + name_bytes + data

def parse_envelope(buf: bytes) -> tuple[str, str | None, bytes]:
    """
    Returns (kind, filename, payload_bytes)
      kind: "text" or "file"
      filename: original file name if kind == "file" else None
    If not wrapped (back-compat), returns ("raw", None, buf).
    """
    if len(buf) < 7 or buf[:4] != ENV_MAGIC:
        return "raw", None, buf
    typ = buf[4]
    name_len = struct.unpack("<H", buf[5:7])[0]
    off = 7
    name = None
    if name_len:
        name = buf[off:off+name_len].decode("utf-8", errors="replace")
    off += name_len
    body = buf[off:]
    return ("text" if typ == 1 else "file"), name, body

# -----------------------------------------------------------------------------#
# Bit helpers & PRP
def bit_chunks(data: bytes, lsb: int) -> Iterable[List[int]]:
    cur = []
    for byte in data:
        for bitpos in range(7, -1, -1):
            cur.append((byte >> bitpos) & 1)
            if len(cur) == lsb:
                yield cur; cur=[]
    if cur:
        while len(cur) < lsb: cur.append(0)
        yield cur

def bits_from_field(value_u8: int, lsb: int) -> List[int]:
    field = value_u8 & ((1<<lsb)-1)
    return [ (field>>i)&1 for i in range(lsb-1,-1,-1) ]

def rotate_left_bits(bits: List[int], lsb: int, kbit_byte: int) -> int:
    if lsb<=0: return 0
    off = (kbit_byte % lsb)
    if off: bits = bits[off:] + bits[:off]
    val = 0
    for b in bits: val = (val<<1)|(b&1)
    return val

def rotate_right(bits: List[int], lsb: int, off: int) -> List[int]:
    if lsb<=0: return bits
    r = off % lsb
    if r==0: return bits
    return bits[-r:] + bits[:-r]

def bits_to_bytes(msb_bits: List[int]) -> bytes:
    out = bytearray(); i=0; n=len(msb_bits)
    while i<n:
        v=0
        for j in range(8):
            v = (v<<1) | (msb_bits[i+j] if (i+j)<n else 0)
        out.append(v); i+=8
    return bytes(out)

def prp_params_from_Kperm(K_perm: bytes, N: int) -> Tuple[int,int]:
    s0 = int.from_bytes(K_perm[:8],"little"); s1=int.from_bytes(K_perm[8:],"little")
    if N<=1: return 0,1
    start = s0 % N; step = (s1 % (N-1)) + 1
    from math import gcd
    while gcd(step, N) != 1:
        step = (step + 1) % N
        if step == 0: step = 1
    return start, step

def prp_index(i: int, start: int, step: int, N: int) -> int: return (start + i*step) % N

def index_to_coord(idx: int, roi: Tuple[int,int,int,int], frame_indices: List[int]) -> Tuple[int,int,int,int]:
    x0,y0,w,h = roi
    chan = idx % 3; idx //= 3
    px = idx % (w*h); frame_off = idx // (w*h)
    y = px // w; x = px % w
    return frame_indices[frame_off], y0+y, x0+x, chan

def embed_header_on_frame(img_rgb: np.ndarray, header_bytes: bytes, lsb: int, roi: tuple[int,int,int,int]):
    """
    Write header sequentially into the chosen LSBs of the ROI on a single frame.
    We use the GREEN channel (index 1) for stability.
    """
    x0, y0, w, h = roi
    need_bits = len(header_bytes) * 8
    # Make a MSB-first bits list
    bits = []
    for b in header_bytes:
        for k in range(7, -1, -1):
            bits.append((b >> k) & 1)
    if w * h * lsb < need_bits:
        raise ValueError("ROI too small for header (increase ROI or LSBs).")

    chan = 1  # G
    mask = (1 << lsb) - 1
    inv = 0xFF ^ mask
    i = 0
    for yy in range(y0, y0 + h):
        row = img_rgb[yy]
        for xx in range(x0, x0 + w):
            if i >= need_bits:
                return
            # take up to lsb bits
            val = 0
            take = min(lsb, need_bits - i)
            for t in range(take):
                val = (val << 1) | bits[i + t]
            # if take < lsb, shift into LSB field (pad zeros at the end)
            if take < lsb:
                val <<= (lsb - take)
            row_x = row[xx]
            row_x[chan] = (row_x[chan] & inv) | val
            i += take

def extract_header_from_frame(img_rgb: np.ndarray, need_bytes: int, lsb: int, roi: tuple[int,int,int,int]) -> bytes:
    """
    Read header sequentially from GREEN channel LSBs in the ROI on a single frame.
    """
    x0, y0, w, h = roi
    need_bits = need_bytes * 8
    chan = 1  # G
    bits = []
    for yy in range(y0, y0 + h):
        row = img_rgb[yy]
        for xx in range(x0, x0 + w):
            if len(bits) >= need_bits:
                break
            field = row[xx][chan] & ((1 << lsb) - 1)
            # append MSB-first bits from the field
            for k in range(lsb - 1, -1, -1):
                if len(bits) >= need_bits:
                    break
                bits.append((field >> k) & 1)
        if len(bits) >= need_bits:
            break
    # pack MSB-first bits to bytes
    out = bytearray()
    for i in range(0, need_bits, 8):
        b = 0
        for k in range(8):
            b = (b << 1) | (bits[i + k] if i + k < need_bits else 0)
        out.append(b)
    return bytes(out)

# -----------------------------------------------------------------------------#
# Video I/O
class VideoReader:
    def __init__(self, path: str):
        self.path = path; self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened(): raise ValueError("Cannot open video: "+path)
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    def get_frame(self, idx: int) -> np.ndarray | None:
        if idx<0 or idx>=self.frames: return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self.cap.read()
        if not ok: return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).copy()
    def close(self):
        try: self.cap.release()
        except: pass

def open_writer_like(reader: VideoReader, out_stem: str):
    """
    Always use PNG sequence + ffmpeg FFV1 (rgb24) so LSBs survive exactly.
    Returns a writer-like object with .write(bgr_frame) and .release().
    codec_name = "FFMPEG_FFV1_SEQ"
    """
    out_path = OUTPUT_DIR / f"{Path(out_stem).name}_stego.mkv"

    class _PngSeqWriter:
        def __init__(self, out_path: Path, fps: float, wh: tuple[int,int]):
            self.out_path = out_path
            self.fps = fps or 25.0
            self.wh = wh
            self.tmpdir = Path(tempfile.mkdtemp(prefix="stego_png_"))
            self.count = 0

        def write(self, bgr_frame):
            # Save exact 8-bit PNG (no compression side-effects on pixel values)
            fn = self.tmpdir / f"frame_{self.count:06d}.png"
            # IMPORTANT: bgr_frame is already BGR; cv2.imwrite expects BGR.
            cv2.imwrite(str(fn), bgr_frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            self.count += 1

        def release(self):
            if self.count == 0:
                shutil.rmtree(self.tmpdir, ignore_errors=True)
                return
            # Mux to FFV1, forcing RGB24 so no colorspace conversion happens
            cmd = [
                "ffmpeg", "-y",
                "-framerate", f"{self.fps}",
                "-i", str(self.tmpdir / "frame_%06d.png"),
                "-c:v", "ffv1", "-pix_fmt", "rgb24",
                str(self.out_path)
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Cleanup temp frames
            shutil.rmtree(self.tmpdir, ignore_errors=True)
            if proc.returncode != 0:
                raise RuntimeError("ffmpeg failed:\n" + proc.stderr.decode("utf-8", errors="ignore"))

    return _PngSeqWriter(out_path, reader.fps, (reader.w, reader.h)), str(out_path), "FFMPEG_FFV1_SEQ"

# ---------- Lossless/losssy helpers (paste near other helpers) ----------
def probe_stream(path: str) -> tuple[str, str]:
    """
    Returns (codec_name, pix_fmt) for the first video stream using ffprobe.
    If ffprobe is missing or fails, returns ('', '').
    """
    try:
        p = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,pix_fmt",
             "-of", "default=noprint_wrappers=1:nokey=1",
             path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if p.returncode != 0:
            return "", ""
        lines = [ln.strip() for ln in p.stdout.splitlines() if ln.strip()]
        # Expect 2 lines: codec_name then pix_fmt
        if len(lines) >= 2:
            return lines[0], lines[1]
        return "", ""
    except Exception:
        return "", ""


def is_safe_for_lsb(codec: str, pix: str) -> bool:
    """
    Return True if this looks like a lossless format that preserves 8-bit RGB/YCbCr exactly.
    We accept ffv1 (rgb24, bgr0, gbrp, etc.) and rawvideo. Everything else is treated as lossy.
    """
    codec = (codec or "").lower()
    pix   = (pix or "").lower()
    if codec in {"ffv1", "rawvideo", "huffyuv", "utvideo"}:
        return True
    # You can whitelist more lossless combos here if you use them.
    return False


def make_view_mp4(from_mkv: str) -> str:
    """
    Create a human-viewable H.264 MP4 copy from the lossless stego MKV.
    NOTE: This is ONLY for watching; decoding from it will break LSBs.
    """
    out_mp4 = str(Path(from_mkv).with_suffix("")) + "_view.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-i", from_mkv,
        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        out_mp4
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        raise RuntimeError("ffmpeg failed while making view MP4:\n" + p.stderr.decode("utf-8", errors="ignore"))
    return out_mp4


# -----------------------------------------------------------------------------#
# UI widgets
class DropLabel(QLabel):
    fileDropped = Signal(str)
    def __init__(self, title: str, exts: set[str] | None):
        super().__init__()
        self.exts = exts
        self.setAcceptDrops(True); self.setAlignment(Qt.AlignCenter)
        self.setObjectName("DropLabel")
        self.setText(f"Drop {title} here\n—or—\nClick to browse")
        self.setStyleSheet("""
        #DropLabel { border: 2px dashed #888; border-radius: 10px; padding: 14px; background: #f7f7f7; color:#444; }
        #DropLabel:hover { background: #eef6ff; }
        """)
    def dragEnterEvent(self,e):
        if e.mimeData().hasUrls():
            p=e.mimeData().urls()[0].toLocalFile()
            if self._ok(p): e.acceptProposedAction()
    def dropEvent(self,e):
        p=e.mimeData().urls()[0].toLocalFile()
        if self._ok(p): self.fileDropped.emit(p)
    def mousePressEvent(self,e):
        if e.button()==Qt.LeftButton:
            dlg = QFileDialog(self); dlg.setFileMode(QFileDialog.ExistingFile)
            if self.exts:
                patt=" ".join(f"*{x}" for x in sorted(self.exts))
                dlg.setNameFilter(f"Supported ({patt})")
            if dlg.exec():
                p = dlg.selectedFiles()[0]
                if self._ok(p): self.fileDropped.emit(p)
    def _ok(self, path: str) -> bool:
        if not os.path.isfile(path): return False
        return True if self.exts is None else (Path(path).suffix.lower() in self.exts)

class FrameView(QLabel):
    roiChanged = Signal(int,int,int,int)
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter); self.setStyleSheet("background:#111;")
        self.setMinimumHeight(320)
        self._img=None; self._pix=None; self._scale=(1.0,1.0); self._offset=(0,0)
        self._rubber=QRubberBand(QRubberBand.Rectangle, self); self._origin=None
    def set_frame(self, img_rgb: np.ndarray | None):
        self._img=img_rgb; self._rubber.hide()
        if img_rgb is None: self.setPixmap(QPixmap()); return
        h,w = img_rgb.shape[:2]
        qimg = QImage(img_rgb.data, w, h, 3*w, QImage.Format.Format_RGB888)
        self._pix = QPixmap.fromImage(qimg)
        scaled = self._pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        sx = scaled.width()/w; sy = scaled.height()/h
        self._scale=(sx,sy); self._offset=((self.width()-scaled.width())//2,(self.height()-scaled.height())//2)
    def resizeEvent(self,e):
        super().resizeEvent(e)
        if self._img is not None: self.set_frame(self._img)
    def mousePressEvent(self,e):
        if self._img is None or e.button()!=Qt.LeftButton: return
        self._rubber.hide(); self._origin=e.pos()
        self._rubber.setGeometry(self._origin.x(), self._origin.y(), 1,1); self._rubber.show()
    def mouseMoveEvent(self,e):
        if self._img is None or not self._rubber.isVisible(): return
        self._rubber.setGeometry(QtCore.QRect(self._origin, e.pos()).normalized())
    def mouseReleaseEvent(self,e):
        if self._img is None or not self._rubber.isVisible(): return
        rect=self._rubber.geometry(); self._rubber.hide()
        x = rect.x()-self._offset[0]; y=rect.y()-self._offset[1]
        x2=rect.right()-self._offset[0]; y2=rect.bottom()-self._offset[1]
        pm=self.pixmap(); 
        if pm is None: return
        pw,ph=pm.width(), pm.height()
        x=max(0,min(x,pw-1)); x2=max(0,min(x2,pw-1))
        y=max(0,min(y,ph-1)); y2=max(0,min(y2,ph-1))
        if x2<=x or y2<=y: return
        sx,sy=self._scale; ix=int(x/sx); iy=int(y/sy); iw=int((x2-x+1)/sx); ih=int((y2-y+1)/sy)
        self.roiChanged.emit(ix,iy,iw,ih)

# -----------------------------------------------------------------------------#
# Encode page (with ROI preview + token)
class PayloadPanel(QWidget):
    def __init__(self, on_changed):
        super().__init__()
        self.payload_path: str | None = None
        v = QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()

        # Text
        t = QWidget(); tl = QVBoxLayout(t)
        self.text_edit = QTextEdit(); self.text_edit.setPlaceholderText("Type or paste text payload here…")
        self.text_info = QLabel("Text bytes: 0")
        self.text_edit.textChanged.connect(self._text_changed)
        tl.addWidget(self.text_edit); tl.addWidget(self.text_info)

        # File
        f = QWidget(); fl = QVBoxLayout(f)
        self.file_drop = DropLabel("a payload file (any type)", None)
        self.file_info = QLabel("No payload file loaded")
        self.file_drop.fileDropped.connect(self._load_file)
        fl.addWidget(self.file_drop); fl.addWidget(self.file_info)

        self.tabs.addTab(t, "Text"); self.tabs.addTab(f, "File")
        self.tabs.currentChanged.connect(on_changed)
        v.addWidget(self.tabs)
        self._on_changed = on_changed

    def mode(self) -> str: return "text" if self.tabs.currentIndex()==0 else "file"
    def payload_bytes(self) -> bytes | None:
        if self.mode()=="text":
            b = self.text_edit.toPlainText().encode("utf-8"); return b if b else None
        if self.payload_path and os.path.isfile(self.payload_path): return open(self.payload_path,"rb").read()
        return None
    def payload_bits(self) -> int | None:
        if self.mode()=="text":
            n=len(self.text_edit.toPlainText().encode("utf-8")); return n*8 if n>0 else None
        if self.payload_path and os.path.isfile(self.payload_path): return os.path.getsize(self.payload_path)*8
        return None
    def _text_changed(self):
        n=len(self.text_edit.toPlainText().encode("utf-8")); self.text_info.setText(f"Text bytes: {n}"); self._on_changed()
    def _load_file(self, path: str):
        try:
            size=os.path.getsize(path); self.payload_path=path
            self.file_info.setText(f"Path: {path}\nSize: {human_bytes(size)}"); self._on_changed()
        except Exception as e:
            QMessageBox.critical(self,"Error",str(e))

class VideoEncodePage(QWidget):
    def __init__(self):
        super().__init__()
        self.reader=None; self.video_path=None; self.cur_frame=0
        self.roi=(0,0,0,0)

        left = QVBoxLayout()

        cov = QGroupBox("Cover Video (MP4, MKV, AVI, MOV)")
        cv = QVBoxLayout()
        self.drop = DropLabel("a video file", SUPPORTED_VIDEO_EXTS)
        self.drop.fileDropped.connect(self.load_video)
        self.info = QLabel("No video loaded"); self.info.setWordWrap(True)
        cv.addWidget(self.drop); cv.addWidget(self.info); cov.setLayout(cv)

        pay = QGroupBox("Payload (Text or File)")
        self.payload_panel = PayloadPanel(self._update_capacity)
        pv = QVBoxLayout(); pv.addWidget(self.payload_panel); pay.setLayout(pv)

        ctrl = QGroupBox("Embedding Controls")
        form = QFormLayout()
        self.lsb = QSlider(Qt.Horizontal); self.lsb.setRange(1,4); self.lsb.setValue(1)
        self.lsb_val = QLabel("1")
        self.lsb.valueChanged.connect(lambda v: self.lsb_val.setText(str(v)))
        self.lsb.valueChanged.connect(self._update_capacity)
        lrow = QHBoxLayout(); lrow.addWidget(self.lsb,1); lrow.addWidget(self.lsb_val,0)
        w_lsb = QWidget(); w_lsb.setLayout(lrow)

        self.key = QLineEdit(); self.key.setPlaceholderText("Enter key / passphrase (required)")

        self.start_frame = QSpinBox(); self.start_frame.setRange(0,0)
        self.len_frames  = QSpinBox();  self.len_frames.setRange(1,1)
        self.frame_step  = QSpinBox();  self.frame_step.setRange(1, 999_999); self.frame_step.setValue(1)
        self.start_frame.valueChanged.connect(self._start_frame_changed)
        self.len_frames.valueChanged.connect(self._update_capacity)
        self.frame_step.valueChanged.connect(self._update_capacity)

        self.start_sec = QDoubleSpinBox(); self.start_sec.setDecimals(3); self.start_sec.setRange(0.0,0.0); self.start_sec.setSingleStep(0.010); self.start_sec.setSuffix(" s")
        self.len_sec   = QDoubleSpinBox(); self.len_sec.setDecimals(3);   self.len_sec.setRange(0.001,0.001); self.len_sec.setSingleStep(0.010); self.len_sec.setSuffix(" s")
        self.start_sec.valueChanged.connect(self._start_sec_changed)
        self.len_sec.valueChanged.connect(self._len_sec_changed)

        form.addRow("Number of LSBs:", w_lsb)
        form.addRow("Key:", self.key)
        form.addRow("Start frame:", self.start_frame); form.addRow("Length (frames):", self.len_frames)
        form.addRow("Frame step:", self.frame_step)
        form.addRow("Start (seconds):", self.start_sec); form.addRow("Length (seconds):", self.len_sec)
        ctrl.setLayout(form)

        ts = QGroupBox("Timestamp (slide to choose start)")
        tsv = QVBoxLayout()
        row = QHBoxLayout()
        self.t_left = QLabel("0:00"); self.t_cur = QLabel("0:00"); self.t_right = QLabel("0:00")
        self.t_cur.setAlignment(Qt.AlignCenter); self.t_cur.setStyleSheet("font-weight:600;")
        row.addWidget(self.t_left); row.addWidget(self.t_cur,1); row.addWidget(self.t_right)
        tsv.addLayout(row)
        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0,0); self.slider.setSingleStep(1); self.slider.setPageStep(30)
        self.slider.valueChanged.connect(self._on_scrub)
        tsv.addWidget(self.slider); ts.setLayout(tsv)

        roi_box = QGroupBox("Selected ROI")
        rv = QVBoxLayout()
        self.roi_label = QLabel("ROI: (x=0, y=0, w=0, h=0)")
        self.roi_thumb = QLabel(); self.roi_thumb.setFixedSize(240, 150); self.roi_thumb.setStyleSheet("background:#222; border:1px solid #444;"); self.roi_thumb.setAlignment(Qt.AlignCenter)
        rv.addWidget(self.roi_label); rv.addWidget(self.roi_thumb); roi_box.setLayout(rv)

        cap = QGroupBox("Capacity")
        self.cap_label = QLabel("Load video + select ROI + add payload."); self.cap_label.setWordWrap(True)
        capv = QVBoxLayout(); capv.addWidget(self.cap_label); cap.setLayout(capv)

        keybox = QGroupBox("Final Key (copy for decoding)")
        keyh = QHBoxLayout()
        self.key_token = QLineEdit(); self.key_token.setReadOnly(True)
        self.btn_copy = QPushButton("Copy"); self.btn_copy.clicked.connect(lambda: (QtGui.QGuiApplication.clipboard().setText(self.key_token.text()), QMessageBox.information(self,"Copied","Final Key copied.")))
        keyh.addWidget(self.key_token); keyh.addWidget(self.btn_copy); keybox.setLayout(keyh)

        self.btn_encode = QPushButton("Encode"); self.btn_encode.clicked.connect(self.on_encode)

        left.addWidget(cov); left.addWidget(pay); left.addWidget(ctrl); left.addWidget(ts); left.addWidget(roi_box); left.addWidget(cap); left.addWidget(keybox); left.addWidget(self.btn_encode); left.addStretch(1)

        # Right
        right = QVBoxLayout()
        vbox = QGroupBox("Video (drag to select ROI)")
        vv = QVBoxLayout()
        self.frame_view = FrameView(); self.frame_view.roiChanged.connect(self._on_roi)
        vv.addWidget(self.frame_view); vbox.setLayout(vv)

        log_box = QGroupBox("Log")
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lv = QVBoxLayout(); lv.addWidget(self.log); log_box.setLayout(lv)

        right.addWidget(vbox); right.addWidget(log_box)

        splitter = QSplitter()
        lw = QWidget(); lw.setLayout(left)
        rw = QWidget(); rw.setLayout(right)
        splitter.addWidget(lw); splitter.addWidget(rw); splitter.setSizes([560, 720])
        root = QVBoxLayout(self); root.addWidget(splitter)

    # ---- helpers
    def load_video(self, path: str):
        try:
            if Path(path).suffix.lower() not in SUPPORTED_VIDEO_EXTS: raise ValueError("Unsupported video format.")
            if self.reader: self.reader.close()
            self.reader = VideoReader(path); self.video_path = path

            dur = self.reader.frames / self.reader.fps if self.reader.fps>0 else 0.0
            self.info.setText(f"Path: {path}\n{self.reader.w}x{self.reader.h} @ {self.reader.fps:.3f} fps, frames={self.reader.frames}, dur={dur:.2f}s")

            self.start_frame.blockSignals(True); self.len_frames.blockSignals(True)
            self.start_frame.setRange(0, max(0, self.reader.frames-1)); self.start_frame.setValue(0)
            self.len_frames.setRange(1, self.reader.frames); self.len_frames.setValue(self.reader.frames)
            self.start_frame.blockSignals(False); self.len_frames.blockSignals(False)

            self.start_sec.blockSignals(True); self.len_sec.blockSignals(True)
            total_dur = dur; min_dur = 1.0 / (self.reader.fps or 1.0)
            self.start_sec.setRange(0.0, max(0.0, total_dur)); self.start_sec.setValue(0.0)
            self.len_sec.setRange(min_dur, max(min_dur, total_dur)); self.len_sec.setValue(max(min_dur, total_dur))
            self.start_sec.blockSignals(False); self.len_sec.blockSignals(False)

            self.slider.blockSignals(True)
            self.slider.setRange(0, max(0, self.reader.frames-1)); self.slider.setValue(0)
            self.slider.blockSignals(False)
            self.t_left.setText("0:00"); self.t_right.setText(fmt_time(self.reader.frames, self.reader.fps)); self.t_cur.setText("0:00")

            self.roi = (0,0,self.reader.w,self.reader.h)
            self._show_frame(0)
            self._update_roi_preview()
            self._update_capacity()
            self.log.append("Loaded video.")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e)); self.log.append(f"[ERROR] {e}")

    def _show_frame(self, idx: int):
        img = self.reader.get_frame(idx) if self.reader else None
        self.cur_frame = idx
        if img is not None:
            self.frame_view.set_frame(img)
            self.t_cur.setText(fmt_time(idx, self.reader.fps))

    def _on_scrub(self, val: int):
        self._show_frame(val)
        if self.reader:
            rate = int(self.reader.fps) or 1
            self.start_frame.blockSignals(True); self.start_frame.setValue(val); self.start_frame.blockSignals(False)
            self.start_sec.blockSignals(True); self.start_sec.setValue(val / rate); self.start_sec.blockSignals(False)
        self._update_roi_preview(); self._update_capacity()

    def _start_frame_changed(self, start_val: int):
        if not self.reader: return
        max_len = max(1, self.reader.frames - start_val)
        self.len_frames.setMaximum(max_len)
        rate = int(self.reader.fps) or 1
        self.start_sec.blockSignals(True); self.len_sec.blockSignals(True)
        self.start_sec.setValue(start_val / rate)
        self.len_sec.setValue(self.len_frames.value() / rate)
        max_len_sec = (self.reader.frames - start_val) / rate
        self.len_sec.setMaximum(max_len_sec if max_len_sec>0 else self.len_sec.minimum())
        self.start_sec.blockSignals(False); self.len_sec.blockSignals(False)
        self.slider.blockSignals(True); self.slider.setValue(start_val); self.slider.blockSignals(False)
        self._show_frame(start_val); self._update_roi_preview(); self._update_capacity()

    def _start_sec_changed(self, s: float):
        if not self.reader: return
        rate = int(self.reader.fps) or 1; total = self.reader.frames
        start_f, len_f = secs_to_frames(s, self.len_sec.value(), rate, total)
        self.start_frame.blockSignals(True); self.len_frames.blockSignals(True)
        self.start_frame.setValue(start_f); self.len_frames.setValue(len_f)
        self.start_frame.blockSignals(False); self.len_frames.blockSignals(False)
        self.slider.blockSignals(True); self.slider.setValue(start_f); self.slider.blockSignals(False)
        self._show_frame(start_f); self._update_roi_preview(); self._update_capacity()

    def _len_sec_changed(self, d: float):
        if not self.reader: return
        rate = int(self.reader.fps) or 1; total = self.reader.frames
        start_f, len_f = secs_to_frames(self.start_sec.value(), d, rate, total)
        self.len_frames.blockSignals(True); self.len_frames.setValue(len_f); self.len_frames.blockSignals(False)
        self._update_capacity()

    def _on_roi(self, x,y,w,h):
        self.roi = (x,y,w,h); self._update_roi_preview(); self._update_capacity()

    def _selected_frames(self) -> List[int]:
        if not self.reader: return []
        start = self.start_frame.value(); length = self.len_frames.value(); step = max(1, self.frame_step.value())
        end = min(self.reader.frames, start + length)
        return list(range(start, end, step))

    def _update_roi_preview(self):
        self.roi_label.setText(f"ROI: (x={self.roi[0]}, y={self.roi[1]}, w={self.roi[2]}, h={self.roi[3]})")
        if not self.reader: self.roi_thumb.clear(); return
        img = self.reader.get_frame(self.cur_frame)
        if img is None or self.roi[2]<=0 or self.roi[3]<=0:
            self.roi_thumb.setText("No ROI"); return
        x,y,w,h = self.roi
        x=max(0,min(x,self.reader.w-1)); y=max(0,min(y,self.reader.h-1))
        w=max(1,min(w,self.reader.w-x)); h=max(1,min(h,self.reader.h-y))
        crop = img[y:y+h, x:x+w]
        if crop.size==0: self.roi_thumb.setText("No ROI"); return
        ch, cw = self.roi_thumb.height(), self.roi_thumb.width()
        scale = min(cw/crop.shape[1], ch/crop.shape[0])
        nw, nh = max(1,int(crop.shape[1]*scale)), max(1,int(crop.shape[0]*scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        qimg = QImage(resized.data, resized.shape[1], resized.shape[0], 3*resized.shape[1], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        canvas = QPixmap(self.roi_thumb.size()); canvas.fill(Qt.black)
        painter = QtGui.QPainter(canvas); painter.drawPixmap((cw-nw)//2, (ch-nh)//2, pix); painter.end()
        self.roi_thumb.setPixmap(canvas)

    def _update_capacity(self):
        if not self.reader:
            self.cap_label.setText("Load video + select ROI + add payload.")
            return
        x,y,w,h = self.roi
        frames = len(self._selected_frames())
        lsb = self.lsb.value()
        if not frames:
            self.cap_label.setText("Adjust Start/Length/Step to select at least 1 frame.")
            return

        safe_cap_bits = safe_payload_capacity_bits(lsb, w, h, frames)  # header-reserved capacity
        payload_bits = self.payload_panel.payload_bits()

        if payload_bits is None:
            self.cap_label.setText(
                f"ROI {w}×{h} @ ({x},{y})  |  Frames {frames} (step={self.frame_step.value()})\n"
                f"LSBs: {lsb}  |  Safe capacity (after header): ≈ {safe_cap_bits} bits"
            )
        else:
            ok = payload_bits <= safe_cap_bits
            self.cap_label.setText(
                f"ROI {w}×{h} @ ({x},{y})  |  Frames {frames} (step={self.frame_step.value()})\n"
                f"LSBs: {lsb}  |  Safe capacity (after header): ≈ {safe_cap_bits} bits\n"
                f"Payload: {payload_bits} bits ({human_bytes(payload_bits//8)})  →  {'OK' if ok else 'Payload too large, require more capacity'}"
            )


    def _log(self, msg: str):
        self.log.append(msg)
    # ---- ENCODE
    def on_encode(self):
        try:
            if not self.reader: raise ValueError("Load a video first.")
            if self.lsb.value() > 2:
                self.log.append("⚠️ LSB > 2 may be visible/fragile; prefer 1–2 for video.")

            key = self.key.text().strip()
            if not key: raise ValueError("Key is required.")

            x0, y0, w, h = self.roi
            if w <= 0 or h <= 0: raise ValueError("Draw an ROI on the video.")

            raw_payload = self.payload_panel.payload_bytes()
            if not raw_payload: raise ValueError("Enter payload text or choose a payload file.")

            # Envelope (keep your behavior)
            is_text = (self.payload_panel.mode() == "text")
            orig_name = None if is_text else (Path(self.payload_panel.payload_path).name if self.payload_panel.payload_path else "payload.bin")
            embed_bytes = make_envelope(is_text, raw_payload, orig_name)

            # Window/ROI
            frames_list = self._selected_frames()
            if not frames_list: raise ValueError("No frames selected (check start/length/step).")
            start_f = frames_list[0]
            length_f = (frames_list[-1] - start_f) + 1
            step = max(1, self.frame_step.value())
            roi = (x0, y0, w, h)
            lsb = self.lsb.value()

            # Key schedule & header
            cover_id = cover_fingerprint(self.reader.path)
            full_salt = canonical_salt_video(lsb, roi, (start_f, length_f, step), cover_id); salt16 = full_salt[:16]
            kd = kdf_from_key(key, salt16)
            K_perm, K_bit, K_check, nonce = kd["K_perm"], kd["K_bit"], kd["K_check"], kd["nonce"]

            header = build_header(1, lsb, roi, len(embed_bytes), cover_id, salt16, nonce, K_check)
            token = make_key_token_video(lsb, roi, start_f, length_f, step, salt16, K_check)
            self.key_token.setText(token)

            # Capacity check (header + payload)
            frames = len(frames_list)
            safe_cap_bits = safe_payload_capacity_bits(lsb, w, h, frames)
            need_payload_bits = len(embed_bytes) * 8

            # Also reject if header itself can’t fit in one frame’s GREEN channel
            if header_fields_needed(lsb) > (w * h):
                raise ValueError("ROI too small for header; increase ROI size or LSBs.")

            if need_payload_bits > safe_cap_bits:
                raise ValueError(
                    f"Payload too large for this ROI/LSB/frame window after reserving header.\n"
                    f"Need {need_payload_bits} bits, have {safe_cap_bits} bits."
                )

            # ---- Reserve header fields on first frame (GREEN channel) ----
            header_bits_needed = len(header) * 8
            header_fields = (header_bits_needed + lsb - 1) // lsb  # how many lsb-fields we use
            reserved = set()  # tuples of (frame_idx, y, x, chan)

            # We write the header sequentially across ROI, GREEN channel, MSB→LSB
            # Reserve exactly the same first 'header_fields' positions we will touch.
            chan_green = 1
            taken = 0
            for yy in range(y0, y0 + h):
                if taken >= header_fields: break
                for xx in range(x0, x0 + w):
                    if taken >= header_fields: break
                    reserved.add((start_f, yy, xx, chan_green))
                    taken += 1

            # ---- Plan PRP positions for PAYLOAD ONLY, skipping the reserved header area ----
            N = frames * w * h * 3
            start_idx, step_perm = prp_params_from_Kperm(K_perm, N)
            payload_fields = (len(embed_bytes) * 8 + lsb - 1) // lsb

            def prp_coord_at(i: int) -> tuple[int,int,int,int]:
                idx = (start_idx + i * step_perm) % N
                return index_to_coord(idx, roi, frames_list)

            chosen_positions: Dict[int, List[Tuple[int,int,int]]] = {}
            i = 0
            picked = 0
            # Iterate until we've collected all payload fields, skipping reserved coords
            # (Cap the loop to a safe upper bound to avoid infinite loops in weird edge cases.)
            safety_cap = N * 3
            while picked < payload_fields and i < safety_cap:
                f, y, x, c = prp_coord_at(i)
                i += 1
                if (f, y, x, c) in reserved:
                    continue
                chosen_positions.setdefault(f, []).append((y, x, c))
                picked += 1

            if picked < payload_fields:
                raise ValueError("Internal error: could not place all payload fields without colliding with header.")

            groups = bit_chunks(embed_bytes, lsb)     # MSB-first groups of size <= lsb
            kbit_byte = K_bit[0]
            mask = (1 << lsb) - 1; invmask = 0xFF ^ mask

            # Open writer: lossless master (FFV1 MKV via PNG sequence)
            stem = str(Path(self.reader.path).with_suffix(""))
            writer, out_path, codec_used = open_writer_like(self.reader, stem)
            self._log(f"Writer codec: {codec_used}")

            # Write frames
            for fidx in range(self.reader.frames):
                frame_rgb = self.reader.get_frame(fidx)  # RGB

                # Header sequential on the first selected frame (GREEN channel)
                if fidx == start_f:
                    embed_header_on_frame(frame_rgb, header, lsb, roi)

                # Payload via PRP across frames/channels (excluding reserved spots)
                if fidx in chosen_positions:
                    for (yy, xx, cc) in chosen_positions[fidx]:
                        try:
                            bits = next(groups)
                        except StopIteration:
                            break
                        v = rotate_left_bits(bits, lsb, kbit_byte)
                        base = frame_rgb[yy, xx, cc] & invmask
                        frame_rgb[yy, xx, cc] = np.uint8(base | v)

                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            writer.release()

            # OPTIONAL: create a lossy "view" MP4 copy for humans
            try:
                view_path = make_view_mp4(out_path)
                self._log(f"View copy (lossy, do NOT decode): {view_path}")
            except Exception as e:
                self._log(f"[WARN] Could not create view MP4: {e}")

            # Self-check header from the lossless MKV
            try:
                vr = VideoReader(out_path)
                img0 = vr.get_frame(start_f)
                hdr2 = extract_header_from_frame(img0, HEADER_TOTAL_LEN, lsb, roi)
                ph2 = parse_header(hdr2, strict_crc=False)
                self._log(f"Self-check header OK: payload_len={ph2.payload_len}, lsb={ph2.lsb}, roi={ph2.roi}")
                vr.close()
            except Exception as e:
                self._log(f"[WARN] Self-check failed: {e}")

            self._show_frame(start_f)
            self.log.append(f"Header bytes={len(header)} salt16={salt16.hex()} bit_rot={kbit_byte % lsb}")
            self.log.append(f"Embedded {len(embed_bytes)*8 + len(header)*8} bits into ROI ({w}x{h}) across {frames} frames (step={step}) @ {lsb} LSB(s).")
            self.log.append(f"Saved stego (lossless master): {out_path}")
            self.log.append(f"Final Key: {token}")

            QMessageBox.information(
                self, "Encode complete",
                "✅ Wrote lossless stego MKV (use this for decoding).\n"
                "A lossy MP4 view copy was also created for watching.\n\n"
                f"Stego MKV:\n{out_path}"
            )

        except Exception as e:
            QMessageBox.critical(self, "Encode error", str(e)); self.log.append(f"[ERROR] {e}")


# -----------------------------------------------------------------------------#
# Compare dialog (optional after decode)
class CompareDialog(QDialog):
    def __init__(self, left_path: str, right_path: str, payload_bytes: bytes, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare: Original vs Stego")
        self.left = VideoReader(left_path)
        self.right = VideoReader(right_path)
        self.timer = QtCore.QTimer(self); self.timer.timeout.connect(self._tick)
        self.playing = True

        top = QHBoxLayout()
        def mk_panel(title):
            g = QGroupBox(title); v = QVBoxLayout()
            view = FrameView(); v.addWidget(view); g.setLayout(v); return g, view
        gL, self.viewL = mk_panel("Original"); gR, self.viewR = mk_panel("Stego")
        top.addWidget(gL); top.addWidget(gR)

        ctrl = QHBoxLayout()
        self.btn_play = QPushButton("Pause"); self.btn_play.clicked.connect(self._toggle)
        self.btn_close = QPushButton("Close"); self.btn_close.clicked.connect(self.accept)
        ctrl.addWidget(self.btn_play); ctrl.addStretch(1); ctrl.addWidget(self.btn_close)

        pay = QGroupBox("Recovered Payload")
        pv = QVBoxLayout()
        self.payload_info = QLabel(f"Size: {len(payload_bytes)} bytes")
        self.payload_view = QTextEdit(); self.payload_view.setReadOnly(True)
        self.btn_save = QPushButton("Save payload as…")
        self.btn_save.clicked.connect(lambda: self._save_payload(payload_bytes))
        try:
            txt = payload_bytes.decode("utf-8")
            self.payload_view.setPlainText(txt)
        except UnicodeDecodeError:
            self.payload_view.setPlainText("(Binary payload – preview not available)")
        pv.addWidget(self.payload_info); pv.addWidget(self.payload_view); pv.addWidget(self.btn_save)
        pay.setLayout(pv)

        root = QVBoxLayout(self)
        root.addLayout(top); root.addLayout(ctrl); root.addWidget(pay)
        self.resize(1200, 800)

        self.idx = 0
        self.timer.start(int(1000.0/ (self.right.fps or 25.0)))
        self._show()

    def _show(self):
        fL = self.left.get_frame(min(self.idx, self.left.frames-1))
        fR = self.right.get_frame(min(self.idx, self.right.frames-1))
        if fL is not None: self.viewL.set_frame(fL)
        if fR is not None: self.viewR.set_frame(fR)

    def _toggle(self):
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")

    def _tick(self):
        if not self.playing: return
        self.idx += 1
        if self.idx >= max(self.left.frames, self.right.frames): self.idx = 0
        self._show()

    def _save_payload(self, data: bytes):
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = OUTPUT_DIR / f"payload_{ts}.bin"
        with open(path, "wb") as f: f.write(data)
        QMessageBox.information(self, "Saved", f"Wrote {len(data)} bytes to:\n{path}")

# -----------------------------------------------------------------------------#
# Decode page — full implementation using token + key
class VideoDecodePage(QWidget):
    def __init__(self):
        super().__init__()
        self.reader: VideoReader | None = None

        left = QVBoxLayout()
        media = QGroupBox("Stego Video (MP4, MKV, AVI, MOV)")
        mv = QVBoxLayout()
        self.drop = DropLabel("a stego video file", SUPPORTED_VIDEO_EXTS)
        self.drop.fileDropped.connect(self.load_video)
        self.info = QLabel("No stego video loaded"); self.info.setWordWrap(True)
        mv.addWidget(self.drop); mv.addWidget(self.info); media.setLayout(mv)

        ctrl = QGroupBox("Controls")
        form = QFormLayout()
        self.key_token = QLineEdit(); self.key_token.setPlaceholderText("Paste Final Key token (stg1:...)")
        self.user_key  = QLineEdit(); self.user_key.setPlaceholderText("Enter the same key/passphrase")
        form.addRow("Final Key:", self.key_token); form.addRow("User Key:", self.user_key)
        ctrl.setLayout(form)

        btns = QHBoxLayout()
        self.btn_inspect = QPushButton("Inspect Header"); self.btn_decode = QPushButton("Decode Payload")
        self.btn_inspect.clicked.connect(self.on_inspect); self.btn_decode.clicked.connect(self.on_decode)
        btns.addWidget(self.btn_inspect); btns.addWidget(self.btn_decode)

        left.addWidget(media); left.addWidget(ctrl); left.addLayout(btns); left.addStretch(1)

        right = QVBoxLayout()
        fbox = QGroupBox("Preview (first frame)")
        fv = QVBoxLayout()
        self.view = FrameView()
        fv.addWidget(self.view); fbox.setLayout(fv)
        right.addWidget(fbox)

        log_box = QGroupBox("Log")
        self.log = QTextEdit(); self.log.setReadOnly(True)
        lv = QVBoxLayout(); lv.addWidget(self.log); log_box.setLayout(lv)
        right.addWidget(log_box)

        splitter = QSplitter()
        lw = QWidget(); l = QVBoxLayout(lw); l.addLayout(left)
        rw = QWidget(); r = QVBoxLayout(rw); r.addLayout(right)
        splitter.addWidget(lw); splitter.addWidget(rw); splitter.setSizes([560, 720])

        root = QVBoxLayout(self); root.addWidget(splitter)

    def load_video(self, path: str):
        try:
            ext = Path(path).suffix.lower()
            if ext not in SUPPORTED_VIDEO_EXTS:
                raise ValueError("Unsupported video format.")

            # Probe stream and refuse likely-lossy inputs (prevents 'Bad header magic')
            codec, pix = probe_stream(path)
            if not is_safe_for_lsb(codec, pix):
                QMessageBox.warning(
                    self, "Likely lossy input",
                    "This video looks lossy (e.g., H.264/HEVC/VP9/AV1). "
                    "Lossy codecs destroy LSBs, so decoding will fail.\n\n"
                    "Please open the *lossless* stego MKV that the Encode tab wrote "
                    "(FFV1 codec)."
                )
                self.log.append(f"[WARN] Refusing likely lossy input. codec={codec or '?'} pix_fmt={pix or '?'}")
                return

            if self.reader: self.reader.close()
            self.reader = VideoReader(path)

            dur = self.reader.frames / self.reader.fps if self.reader.fps > 0 else 0.0
            self.info.setText(
                f"Path: {path}\n"
                f"{self.reader.w}x{self.reader.h} @ {self.reader.fps:.3f} fps, "
                f"frames={self.reader.frames}, dur={dur:.2f}s\n"
                f"codec={codec or '?'} pix_fmt={pix or '?'}"
            )
            img = self.reader.get_frame(0)
            self.view.set_frame(img)
            self.log.append("Loaded stego video (lossless).")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            self.log.append(f"[ERROR] {e}")


    def _derive(self, token: str, user_key: str):
        info = parse_key_token(token)
        kd = kdf_from_key(user_key, info["salt16"])
        if info["kcheck4"] != kd["K_check"]:
            raise ValueError("Wrong passphrase or tampered Final Key (K_check mismatch)")
        return info, kd

    def _plan_positions(self, info, kd, params_only=False, bits_total=None):
        lsb = info["lsb"]; x0,y0,w,h = info["roi"]; start_f = info["start_f"]; len_f = info["len_f"]; step = max(1, info.get("step",1))
        frames_list = list(range(start_f, min(start_f+len_f, self.reader.frames), step))
        if params_only:
            N = len(frames_list) * w * h * 3
            return N, (x0,y0,w,h), frames_list, lsb
        N = len(frames_list) * w * h * 3
        start_idx, step_perm = prp_params_from_Kperm(kd["K_perm"], N)
        fields_needed = (bits_total + lsb - 1) // lsb if bits_total is not None else None
        return N, (x0,y0,w,h), frames_list, lsb, start_idx, step_perm, fields_needed

    def _extract_bits(self, info, kd, total_bits_needed: int) -> List[int]:
        # Same PRP parameters as encode
        N, roi, frames_list, lsb, start_idx, step_perm, fields_needed = self._plan_positions(
            info, kd, bits_total=total_bits_needed
        )
        x0, y0, w, h = roi
        start_f = info["start_f"]

        # --- Build the exact same reserved set the encoder used for the header ---
        # Header is sequential over GREEN on the first selected frame
        header_fields = (HEADER_TOTAL_LEN * 8 + lsb - 1) // lsb
        reserved = set()
        chan_green = 1
        taken = 0
        for yy in range(y0, y0 + h):
            if taken >= header_fields:
                break
            for xx in range(x0, x0 + w):
                if taken >= header_fields:
                    break
                reserved.add((start_f, yy, xx, chan_green))
                taken += 1

        # --- Plan payload positions via PRP, skipping reserved ---
        def prp_coord_at(i: int) -> tuple[int,int,int,int]:
            idx = (start_idx + i * step_perm) % N
            return index_to_coord(idx, roi, frames_list)

        chosen_positions: Dict[int, List[Tuple[int,int,int]]] = {}
        picked = 0
        i = 0
        safety_cap = N * 3
        while picked < fields_needed and i < safety_cap:
            f, y, x, c = prp_coord_at(i)
            i += 1
            if (f, y, x, c) in reserved:
                continue
            chosen_positions.setdefault(f, []).append((y, x, c))
            picked += 1

        if picked < fields_needed:
            raise ValueError("Internal error: could not plan all payload fields without colliding with header.")

        # --- Extract bits in the same order, undoing the bit rotation ---
        bits_out: List[int] = []
        rot = kd["K_bit"][0] % lsb if lsb > 0 else 0
        for fidx in range(self.reader.frames):
            if fidx not in chosen_positions:
                continue
            frame = self.reader.get_frame(fidx)  # RGB
            for (yy, xx, cc) in chosen_positions[fidx]:
                field_bits = bits_from_field(int(frame[yy, xx, cc]), lsb)
                if rot:
                    field_bits = rotate_right(field_bits, lsb, rot)
                bits_out.extend(field_bits)
                if len(bits_out) >= total_bits_needed:
                    return bits_out[:total_bits_needed]
        return bits_out


    def on_inspect(self):
        try:
            if not self.reader: raise ValueError("Load a stego video first.")
            token = self.key_token.text().strip(); user_key = self.user_key.text().strip()
            if not token or not user_key: raise ValueError("Provide both Final Key token and the user key.")

            info, kd = self._derive(token, user_key)
            lsb = info["lsb"]; roi = info["roi"]; start_f = info["start_f"]

            # Read header deterministically from the first selected frame
            img0 = self.reader.get_frame(start_f)
            header_bytes = extract_header_from_frame(img0, HEADER_TOTAL_LEN, lsb, roi)
            ph = parse_header(header_bytes, strict_crc=False)

            ok_ctx = (ph.lsb == lsb and ph.roi == roi)
            self.log.append(
                f"Header OK: version={ph.version} payload_len={ph.payload_len} "
                f"lsb={ph.lsb} roi={ph.roi} salt16={ph.salt16.hex()} crc32=0x{ph.crc32:08x}"
            )
            self.log.append("Context matches token: " + ("YES" if ok_ctx else "NO (mismatch)"))
            QMessageBox.information(self, "Inspect", "Header parsed successfully. See log for details.")
        except Exception as e:
            QMessageBox.critical(self, "Inspect error", str(e)); self.log.append(f"[ERROR] {e}")


    def on_decode(self):
        try:
            if not self.reader: raise ValueError("Load a stego video first.")
            token = self.key_token.text().strip(); user_key = self.user_key.text().strip()
            if not token or not user_key: raise ValueError("Provide both Final Key token and the user key.")

            info, kd = self._derive(token, user_key)
            lsb = info["lsb"]; roi = info["roi"]; start_f = info["start_f"]

            # 1) Header (deterministic from first frame)
            img0 = self.reader.get_frame(start_f)
            header_bytes = extract_header_from_frame(img0, HEADER_TOTAL_LEN, lsb, roi)
            ph = parse_header(header_bytes)  # strict CRC
            if ph.kcheck4 != kd["K_check"]:
                raise ValueError("Wrong key/token (K_check mismatch)")
            if ph.lsb != lsb or ph.roi != roi:
                raise ValueError("Token/controls do not match embedded header")

            # 2) Payload via PRP
            need_bits = ph.payload_len * 8
            payload_bits = self._extract_bits(info, kd, need_bits)
            payload_bytes = bits_to_bytes(payload_bits)

            # 3) Unwrap envelope -> decide action
            kind, fname, body = parse_envelope(payload_bytes)

            ts = time.strftime("%Y%m%d-%H%M%S")
            base = Path(self.reader.path).stem

            if kind == "text":
                # print the UTF-8 text to the log (and also save a txt for convenience if you want)
                try:
                    txt = body.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    # If encode-time said "text" but it isn't valid UTF-8, just show hex preview and save .bin
                    hex_preview = body[:48].hex(" ")
                    self.log.append(f"[WARN] Text payload is not valid UTF-8. First bytes: {hex_preview} …")
                    out_path = OUTPUT_DIR / f"{base}_payload_{ts}.bin"
                    with open(out_path, "wb") as f: f.write(body)
                    QMessageBox.information(self, "Decode", f"Saved raw bytes to:\n{out_path}")
                    return

                self.log.append("---- TEXT PAYLOAD START ----")
                self.log.append(txt)
                self.log.append("---- TEXT PAYLOAD END ----")
                QMessageBox.information(self, "Decode", "Text payload printed to the log.")

            elif kind == "file":
                safe_name = _sanitize_filename(fname or "payload.bin")
                # avoid clobbering existing file
                out_path = OUTPUT_DIR / safe_name
                if out_path.exists():
                    out_path = OUTPUT_DIR / f"{out_path.stem}_{ts}{out_path.suffix}"
                with open(out_path, "wb") as f:
                    f.write(body)
                self.log.append(f"Saved file payload to: {out_path} ({len(body)} bytes)")
                QMessageBox.information(self, "Decode", f"File saved:\n{out_path}")

            else:
                # Back-compat path (older stegos without envelope)
                try:
                    txt = payload_bytes.decode("utf-8")
                    self.log.append("---- TEXT PAYLOAD (legacy) ----")
                    self.log.append(txt)
                    self.log.append("---- END ----")
                    QMessageBox.information(self, "Decode", "Text payload (legacy) printed to the log.")
                except UnicodeDecodeError:
                    out_path = OUTPUT_DIR / f"{base}_payload_{ts}.bin"
                    with open(out_path, "wb") as f: f.write(payload_bytes)
                    self.log.append(f"Saved binary payload to: {out_path} ({len(payload_bytes)} bytes)")
                    QMessageBox.information(self, "Decode", f"Binary payload saved:\n{out_path}")

            # Optional compare dialog
            orig_path, _ = QFileDialog.getOpenFileName(self, "Select original/cover video (optional for comparison)")
            if orig_path:
                dlg = CompareDialog(orig_path, self.reader.path, payload_bytes, self)
                dlg.exec()

        except Exception as e:
            QMessageBox.critical(self, "Decode error", str(e)); self.log.append(f"[ERROR] {e}")

class VideoSuite(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        tabs = QTabWidget()
        tabs.addTab(VideoEncodePage(), "Encode")
        tabs.addTab(VideoDecodePage(), "Decode")

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)


# -----------------------------------------------------------------------------#
# Window / main
class VideoStegoWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video Steganography")
        tabs = QTabWidget()
        tabs.addTab(VideoEncodePage(), "Encode")
        tabs.addTab(VideoDecodePage(), "Decode")
        self.setCentralWidget(tabs)
        self.resize(1280, 820)

if __name__ == "__main__":
    app = QApplication([])
    win = VideoStegoWindow()
    win.show()
    app.exec()
