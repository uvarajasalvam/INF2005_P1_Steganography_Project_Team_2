# image_ui.py
import os, io, zlib, hmac, hashlib, struct, base64
from pathlib import Path

from numpy.random import PCG64, Generator
from PySide6.QtGui import QPixmap, QMovie
from PIL import Image

from PySide6.QtCore import Qt, QRect, Signal
import PySide6.QtCore as QtCore
import PySide6.QtGui as QtGui
import PySide6.QtWidgets as QtWidgets
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QGroupBox,
    QFileDialog, QLineEdit, QFormLayout, QTabWidget, QSplitter, QFrame,
    QMessageBox, QTextEdit, QRubberBand, QSlider
)

# ---------- File types ----------
IMAGE_EXTS = {".bmp", ".png", ".gif", ".jpg", ".jpeg"}

# ---------- Utils (shared) ----------
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

def rng_from_16_bytes(k16: bytes) -> Generator:
    seed_int = int.from_bytes(k16, "little", signed=False)
    return Generator(PCG64(seed_int))

def build_header(version: int, lsb: int, roi_xywh: tuple[int,int,int,int],
                 payload_len: int, cover_fp16: bytes, salt16: bytes,
                 nonce12: bytes, kcheck4: bytes) -> bytes:
    magic = b"STG1"; flags = 0; pad = 0
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

def parse_header(buf: bytes) -> dict:
    if len(buf) < 84 or buf[:4] != b"STG1":
        raise ValueError("Invalid or missing header.")
    version, flags, lsb, pad = struct.unpack("<BBBB", buf[4:8])
    x0, y0, w, h = struct.unpack("<IIII", buf[8:24])
    payload_len, = struct.unpack("<Q", buf[24:32])
    cover_fp16 = buf[32:48]; salt16 = buf[48:64]
    nonce12 = buf[64:76]; kcheck4 = buf[76:80]
    crc_stored, = struct.unpack("<I", buf[80:84])
    crc_calc = zlib.crc32(buf[:80]) & 0xFFFFFFFF
    if crc_calc != crc_stored:
        raise ValueError("Header CRC mismatch.")
    return {
        "version": version, "flags": flags, "lsb": lsb, "roi": (x0,y0,w,h),
        "payload_len": payload_len, "cover_fp16": cover_fp16,
        "salt16": salt16, "nonce12": nonce12, "kcheck4": kcheck4
    }

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

# ---------- Stego helpers ----------
def _ensure_8bit_mode(pil_img: Image.Image) -> tuple[Image.Image, int, str]:
    """
    Ensure image is 8-bit per channel. Return (converted_img, channels, mode).
    We support L, RGB, RGBA. Convert other modes to RGB.
    """
    mode = pil_img.mode
    if mode == "L":
        return pil_img, 1, "L"
    if mode == "RGB":
        return pil_img, 3, "RGB"
    if mode == "RGBA":
        return pil_img, 4, "RGBA"
    # Palette or others → RGB
    return pil_img.convert("RGB"), 3, "RGB"

def _img_bytes_and_geometry(img: Image.Image) -> tuple[bytearray, int, int, int]:
    """Return (mutable_bytes, width, height, channels)."""
    conv, ch, _ = _ensure_8bit_mode(img)
    w, h = conv.size
    return bytearray(conv.tobytes()), w, h, ch

def _pixel_byte_index(x: int, y: int, c: int, width: int, channels: int) -> int:
    return (y*width + x)*channels + c

def _set_bit(value: int, plane: int, bit: int) -> int:
    mask = 1 << plane
    value &= ~mask
    if bit: value |= mask
    return value

def _get_bit(value: int, plane: int) -> int:
    return (value >> plane) & 1

def _make_perm(total_slots: int, K_perm: bytes) -> list[int]:
    """Permutation over slot indices [0..total_slots-1]."""
    rng = rng_from_16_bytes(K_perm)
    return rng.permutation(total_slots).tolist()

def _hmac_keystream(key32: bytes, nonce12: bytes, nbytes: int) -> bytes:
    """
    Deterministic keystream from stdlib only:
    KS[i*32:(i+1)*32] = HMAC-SHA256(key32, nonce||uint64_le(counter))
    """
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        ctr = struct.pack("<Q", counter)
        out.extend(hmac.new(key32, nonce12 + ctr, hashlib.sha256).digest())
        counter += 1
    return bytes(out[:nbytes])

def _xor(data: bytes, stream: bytes) -> bytes:
    return bytes(a ^ b for a,b in zip(data, stream))

def _is_mostly_text(b: bytes) -> bool:
    # Heuristic: try UTF-8, then check printable ratio
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return False
    # consider it text if most chars are printable/whitespace
    printable = sum(ch.isprintable() or ch.isspace() for ch in s)
    return (printable / max(1, len(s))) > 0.95

def guess_image_or_text_ext(data: bytes) -> tuple[str, str, bool]:
    """
    Return (ext, label, is_text). Prioritize image signatures over text.
    """
    sig = data[:16]

    # --- images (magic bytes) ---
    if sig.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "PNG image", False
    if sig.startswith(b"\xff\xd8\xff"):
        return ".jpg", "JPEG image", False
    if sig.startswith(b"GIF87a") or sig.startswith(b"GIF89a"):
        return ".gif", "GIF image", False
    if sig.startswith(b"BM"):
        return ".bmp", "BMP image", False
    if sig.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP":
        return ".webp", "WebP image", False
    if sig.startswith(b"II*\x00") or sig.startswith(b"MM\x00*"):
        return ".tif", "TIFF image", False
    if len(sig) >= 4 and sig[:4] == b"\x00\x00\x01\x00":
        return ".ico", "ICO image", False

    # SVG (XML/text). Check early bytes for '<svg' or xml with svg
    head = data[:256].lstrip()
    if head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head[:256]):
        return ".svg", "SVG image", True  # text-based image

    # --- text (UTF-8) ---
    if _is_mostly_text(data):
        return ".txt", "UTF-8 text", True

    # unknown
    return ".bin", "Unknown binary", False

# ---------- Widgets ----------
class ImageView(QtWidgets.QFrame):
    roiSelected = Signal(int, int, int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self._pix = QPixmap()
        self._movie: QMovie | None = None # for animated GIFs
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._origin = None
        self._display_rect = QRect()
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Plain)

    # accept either QPixmap or QMovie
    def setImage(self, media):
        # stop any existing movie
        if getattr(self, "_movie", None):
            try:
                self._movie.frameChanged.disconnect(self.update)
            except Exception:
                pass
            self._movie.stop()
            self._movie = None

        if isinstance(media, QMovie):
            self._movie = media
            self._pix = QPixmap()

            # Repaint whenever a new frame is available
            self._movie.frameChanged.connect(self.update)

            # Start playback and try to ensure an initial frame
            self._movie.start()
            # Sometimes the very first currentPixmap() is still null; try to force frame 0
            if self._movie.currentPixmap().isNull():
                self._movie.jumpToFrame(0)

        else:
            # Assume QPixmap (or None)
            self._pix = media if media and not media.isNull() else QPixmap()

        self.update()

    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        p.fillRect(self.rect(), Qt.white)

        pix = self._current_pixmap()          # <-- use helper
        if not self._has_image():             # <-- use helper
            p.setPen(Qt.gray)
            p.drawText(self.rect(), Qt.AlignCenter, "Drop or load an image")
            return

        w = self.width(); h = self.height()
        pw = pix.width(); ph = pix.height()
        scale = min(w/pw, h/ph)
        dw = int(pw*scale); dh = int(ph*scale)
        x0 = (w - dw)//2; y0 = (h - dh)//2
        target = QtCore.QRect(x0, y0, dw, dh)
        self._display_rect = target
        p.drawPixmap(target, pix)

    def mousePressEvent(self, e):
        if not self._has_image():             # <-- was: if self._pix.isNull()
            return
        if e.button() == Qt.LeftButton:
            self._origin = e.position().toPoint()
            self._rubber.setGeometry(QRect(self._origin, QtCore.QSize()))
            self._rubber.show()

    def mouseMoveEvent(self, e):
        if self._rubber.isVisible() and self._origin is not None:
            rect = QRect(self._origin, e.position().toPoint()).normalized()
            self._rubber.setGeometry(rect)

    def mouseReleaseEvent(self, e):
        if not self._rubber.isVisible():
            return
        self._rubber.hide()
        if self._origin is None or not self._has_image():  # <-- ensure image exists
            return
        rect = self._rubber.geometry()
        self._origin = None
        if rect.width() <= 0 or rect.height() <= 0:
            return
        if self._display_rect.width() == 0 or self._display_rect.height() == 0:
            return

        # Map selection back to source pixels using the current frame size
        pix = self._current_pixmap()
        pw, ph = pix.width(), pix.height()
        sx = pw / self._display_rect.width()
        sy = ph / self._display_rect.height()
        x0 = int((rect.left() - self._display_rect.left()) * sx)
        y0 = int((rect.top()  - self._display_rect.top())  * sy)
        w  = int(rect.width()  * sx)
        h  = int(rect.height() * sy)
        x0 = max(0, min(x0, pw - 1))
        y0 = max(0, min(y0, ph - 1))
        w = max(1, min(w, pw - x0))
        h = max(1, min(h, ph - y0))
        self.roiSelected.emit(x0, y0, w, h)
        
    def _current_pixmap(self) -> QPixmap:
        return self._movie.currentPixmap() if self._movie else self._pix

    def _has_image(self) -> bool:
        pix = self._current_pixmap()
        return bool(pix) and not pix.isNull()
    
class ZoomImageView(QtWidgets.QGraphicsView):
    """
    Zoomable + pannable image view.
    - Mouse wheel: zoom in/out (cursor-centered)
    - Left-drag: pan (hand tool)
    - Double-click: fit to window
    - Public helpers: set_pixmap(QPixmap), set_movie(QMovie), fit_to_window(), set_zoom(1.0)
    - Emits viewChanged(scale, center_x, center_y) to allow syncing two views.
    """
    viewChanged = QtCore.Signal(float, float, float)  # scale, center.x, center.y

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        self.setBackgroundBrush(Qt.white)
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorViewCenter)

        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixitem = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pixitem)

        self._movie: QMovie | None = None
        self._base_pix = QPixmap()   # current frame in pixmap form
        self._scale = 1.0

    # ---------- Public API ----------
    def setImage(self, media):
        """Accept QPixmap or QMovie (same signature as your ImageView)."""
        # Stop old movie
        if self._movie:
            try:
                self._movie.frameChanged.disconnect(self._on_movie_frame)
            except Exception:
                pass
            self._movie.stop()
            self._movie = None

        if isinstance(media, QMovie):
            self._movie = media
            self._movie.frameChanged.connect(self._on_movie_frame)
            self._movie.start()
            if self._movie.currentPixmap().isNull():
                self._movie.jumpToFrame(0)
            self._on_movie_frame()
        else:
            # Assume QPixmap (or None)
            self.set_pixmap(media if media and not media.isNull() else QPixmap())

    def set_pixmap(self, pix: QPixmap):
        self._base_pix = pix if pix and not pix.isNull() else QPixmap()
        self._pixitem.setPixmap(self._base_pix)
        self._reset_view()

    def set_movie(self, movie: QMovie):
        self.setImage(movie)

    def fit_to_window(self):
        if self._base_pix.isNull():
            return
        self._scale = 1.0
        self.resetTransform()
        # Fit keeping aspect ratio
        br = self._pixitem.boundingRect()
        if br.isNull():
            return
        self.fitInView(br, Qt.KeepAspectRatio)
        # Record resulting scale from transform
        self._scale = self.transform().m11()
        self._emit_view_changed()

    def set_zoom(self, scale: float):
        if scale <= 0 or self._base_pix.isNull():
            return
        # Reset then apply scale so we don't accumulate floating-point error
        center = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(center)
        self._scale = scale
        self._emit_view_changed()

    def current_zoom(self) -> float:
        return self._scale

    # ---------- Internals ----------
    def _on_movie_frame(self, *_):
        if not self._movie:
            return
        frame = self._movie.currentPixmap()
        if not frame.isNull():
            self._base_pix = frame
            self._pixitem.setPixmap(self._base_pix)
            # Keep the same transform/scale, but ensure the scene rect fits
            self._scene.setSceneRect(self._pixitem.boundingRect())

    def _reset_view(self):
        self._scene.setSceneRect(self._pixitem.boundingRect())
        self.fit_to_window()

    def wheelEvent(self, event: QtGui.QWheelEvent):
        if self._base_pix.isNull():
            return
        # Typical zoom step
        step = 1.15 if event.angleDelta().y() > 0 else (1/1.15)
        new_scale = max(0.05, min(50.0, self._scale * step))
        # Apply new scale (cursor-centered thanks to AnchorUnderMouse)
        self.set_zoom(new_scale)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        # Keep "fit" feeling when window resizes if we're approximately fitted
        super().resizeEvent(event)
        if self._base_pix.isNull():
            return
        # Heuristic: if scale is within 3% of a perfect fit, re-fit on resize
        br = self._pixitem.boundingRect()
        if br.isNull():
            return
        view_rect = self.viewport().rect()
        if not view_rect.isValid():
            return
        # Compute best fit scale
        view_w = max(1, view_rect.width())
        view_h = max(1, view_rect.height())
        fit_scale = min(view_w / max(1, br.width()), view_h / max(1, br.height()))
        if abs(self._scale - fit_scale) / fit_scale < 0.03:
            self.fit_to_window()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        super().mouseReleaseEvent(event)
        self._emit_view_changed()

    def _emit_view_changed(self):
        c = self.mapToScene(self.viewport().rect().center())
        self.viewChanged.emit(self._scale, c.x(), c.y())
class DecodePreviewDialog(QtWidgets.QDialog):
    """
    Top row: Stego (left) ⟷ Cover (right) for instant visual comparison.
    Bottom row: Payload preview (text/image/other) with 'Open folder' button.
    """
    def __init__(self, stego_path: str, payload_path: str, payload_label: str, is_text: bool, cover_path: str | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Decode Preview")
        self.resize(1100, 750)

        self._link_views = True  # whether to sync zoom/pan between stego & cover
        self.stego_path = stego_path
        self.payload_path = payload_path
        self.payload_label = payload_label
        self.is_text = is_text

        # ---------- TOP: side-by-side comparison ----------
        top = QtWidgets.QSplitter(Qt.Horizontal)

        # Stego panel (left)
        stego_panel = QtWidgets.QWidget()
        sl = QtWidgets.QVBoxLayout(stego_panel); sl.setContentsMargins(0,0,0,0)

        # CHANGED: zoomable view
        self.stego_view = ZoomImageView()
        self._load_any_image(self.stego_view, stego_path)
        sl.addWidget(self.stego_view, 1)

        # small toolbar under stego
        sbar = QtWidgets.QHBoxLayout()
        sbar.addWidget(QtWidgets.QLabel(f"Stego: {Path(stego_path).name}"))
        sbar.addStretch(1)
        self.btn_fit_left = QtWidgets.QPushButton("Fit")
        self.btn_100_left = QtWidgets.QPushButton("100%")
        self.btn_zoom_in_left = QtWidgets.QPushButton("+")
        self.btn_zoom_out_left = QtWidgets.QPushButton("–")
        for b in (self.btn_fit_left, self.btn_100_left, self.btn_zoom_in_left, self.btn_zoom_out_left):
            sbar.addWidget(b)
        sl.addLayout(sbar)

        # Cover panel (right)
        cover_panel = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(cover_panel); cl.setContentsMargins(0,0,0,0)

        # CHANGED: zoomable view
        self.cover_view = ZoomImageView()
        cl.addWidget(self.cover_view, 1)

        cbar = QtWidgets.QHBoxLayout()
        cbar.addWidget(QtWidgets.QLabel("Cover:"))
        cbar.addStretch(1)
        self.link_chk = QtWidgets.QCheckBox("Link zoom/pan")
        self.link_chk.setChecked(True)
        self.btn_fit_right = QtWidgets.QPushButton("Fit")
        self.btn_100_right = QtWidgets.QPushButton("100%")
        self.btn_zoom_in_right = QtWidgets.QPushButton("+")
        self.btn_zoom_out_right = QtWidgets.QPushButton("–")
        cbar.addWidget(self.link_chk)
        for b in (self.btn_fit_right, self.btn_100_right, self.btn_zoom_in_right, self.btn_zoom_out_right):
            cbar.addWidget(b)
        cl.addLayout(cbar)

        top.addWidget(stego_panel); top.addWidget(cover_panel); top.setSizes([550, 550])

        # ---------- BOTTOM: payload preview ----------
        payload_box = QtWidgets.QGroupBox(f"Payload ({payload_label})")
        pl = QtWidgets.QVBoxLayout(payload_box)

        self.payload_stack = QtWidgets.QStackedWidget()
        # idx 0: text
        self.payload_text = QtWidgets.QTextEdit(); self.payload_text.setReadOnly(True)
        # idx 1: image
        self.payload_img = ImageView()
        # idx 2: generic info
        self.payload_info = QtWidgets.QLabel(); self.payload_info.setWordWrap(True)

        self.payload_stack.addWidget(self.payload_text)  # 0
        self.payload_stack.addWidget(self.payload_img)   # 1
        self.payload_stack.addWidget(self.payload_info)  # 2

        pl.addWidget(self.payload_stack)
        btns = QtWidgets.QHBoxLayout()
        self.btn_open_folder = QtWidgets.QPushButton("Open payload folder")
        self.btn_open_folder.clicked.connect(self._open_payload_folder)
        btns.addStretch(1); btns.addWidget(self.btn_open_folder)
        pl.addLayout(btns)

        self._load_payload_preview()

        # ---------- Main layout ----------
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(top, 1)
        layout.addWidget(payload_box, 1)

        # Auto-load cover if provided
        if cover_path and Path(cover_path).exists():
            self._load_any_image(self.cover_view, cover_path)

        # --- Hook up toolbar buttons ---
        self.btn_fit_left.clicked.connect(self.stego_view.fit_to_window)
        self.btn_fit_right.clicked.connect(self.cover_view.fit_to_window)
        self.btn_100_left.clicked.connect(lambda: self.stego_view.set_zoom(1.0))
        self.btn_100_right.clicked.connect(lambda: self.cover_view.set_zoom(1.0))
        self.btn_zoom_in_left.clicked.connect(lambda: self._step_zoom(self.stego_view, 1.15))
        self.btn_zoom_out_left.clicked.connect(lambda: self._step_zoom(self.stego_view, 1/1.15))
        self.btn_zoom_in_right.clicked.connect(lambda: self._step_zoom(self.cover_view, 1.15))
        self.btn_zoom_out_right.clicked.connect(lambda: self._step_zoom(self.cover_view, 1/1.15))
        self.link_chk.toggled.connect(self._set_linked)

        # --- Sync the two views when "linked" ---
        self.stego_view.viewChanged.connect(lambda s, x, y: self._maybe_sync(self.cover_view, s, x, y, source="left"))
        self.cover_view.viewChanged.connect(lambda s, x, y: self._maybe_sync(self.stego_view, s, x, y, source="right"))

    # -------- helpers --------
    def _choose_cover(self):
        dlg = QtWidgets.QFileDialog(self, "Choose Cover Image", str(Path(self.stego_path).parent))
        dlg.setFileMode(QtWidgets.QFileDialog.ExistingFile)
        dlg.setNameFilter("Images (*.bmp *.png *.gif *.jpg *.jpeg *.webp *.tif *.tiff *.ico)")
        if dlg.exec():
            p = dlg.selectedFiles()[0]
            self._load_any_image(self.cover_view, p)

    def _load_any_image(self, view, path: str):
        ext = Path(path).suffix.lower()
        if ext == ".gif":
            mv = QMovie(path)
            view.setImage(mv)
        else:
            qpix = QPixmap(path)
            view.setImage(qpix)


    def _load_payload_preview(self):
        # Text
        if self.is_text and self.payload_path.lower().endswith(".txt"):
            try:
                txt = open(self.payload_path, "r", encoding="utf-8").read()
                self.payload_text.setPlainText(txt)
                self.payload_stack.setCurrentIndex(0)
                return
            except Exception as e:
                self.payload_info.setText(f"Could not display text: {e}\nFile: {self.payload_path}")
                self.payload_stack.setCurrentIndex(2)
                return

        # Image preview
        if Path(self.payload_path).suffix.lower() in {".png",".jpg",".jpeg",".gif",".bmp",".webp",".tif",".tiff",".ico"}:
            try:
                self._load_any_image(self.payload_img, self.payload_path)
                self.payload_stack.setCurrentIndex(1)
                return
            except Exception:
                pass

        # Fallback info
        size = os.path.getsize(self.payload_path) if os.path.exists(self.payload_path) else 0
        self.payload_info.setText(
            f"Saved payload: {self.payload_path}\n"
            f"Type: {self.payload_label}\n"
            f"Size: {human_bytes(size)}\n"
            f"(Preview not available.)"
        )
        self.payload_stack.setCurrentIndex(2)

    def _open_payload_folder(self):
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(Path(self.payload_path).parent)))
        
    def _step_zoom(self, view: 'ZoomImageView', factor: float):
        view.set_zoom(max(0.05, min(50.0, view.current_zoom() * factor)))

    def _set_linked(self, on: bool):
        self._link_views = on

    def _maybe_sync(self, target: 'ZoomImageView', scale: float, cx: float, cy: float, source: str):
        if not self._link_views:
            return
        # Apply the same zoom and center to the other view
        target.set_zoom(scale)
        target.centerOn(QtCore.QPointF(cx, cy))


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
            #DropLabel { border: 2px dashed #888; border-radius: 12px; padding: 16px;
                         color: #444; font-size: 14px; background: #fafafa; }
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

class PayloadPanel(QWidget):
    def __init__(self, on_changed):
        super().__init__()
        self.payload_path: str | None = None
        box = QVBoxLayout(self)
        self.tabs = QTabWidget()
        # Text
        self.text_edit = QTextEdit()
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


# ---------- IMAGE ENCODE ----------
class ImageEncodeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cover_path: str | None = None
        self.cover_qpix: QPixmap | None = None
        self.img_info = None
        self.roi_img = None

        left = QVBoxLayout()
        cov_box = QGroupBox("Cover Image")
        cv = QVBoxLayout()
        self.cover_drop = DropLabel("an image (.bmp/.png/.gif/.jpg/.jpeg)", IMAGE_EXTS)
        self.cover_drop.fileDropped.connect(self.load_cover)
        self.cover_info = QLabel("No cover loaded"); self.cover_info.setWordWrap(True)
        cv.addWidget(self.cover_drop); cv.addWidget(self.cover_info)
        cov_box.setLayout(cv)

        pay_box = QGroupBox("Payload (Text or File)")
        self.payload_panel = PayloadPanel(self.update_capacity_label)
        pay_v = QVBoxLayout(); pay_v.addWidget(self.payload_panel); pay_box.setLayout(pay_v)

        ctrl = QGroupBox("Embedding Controls")
        form = QFormLayout()
        self.lsb_slider = QSlider(Qt.Horizontal); self.lsb_slider.setRange(1,8); self.lsb_slider.setValue(1)
        self.lsb_value = QLabel("1")
        self.lsb_slider.valueChanged.connect(lambda v: self.lsb_value.setText(str(v)))
        self.lsb_slider.valueChanged.connect(self.update_capacity_label)
        self.key_edit = QLineEdit(); self.key_edit.setPlaceholderText("Enter numeric/passphrase key (required)")
        lrow = QHBoxLayout(); lrow.addWidget(self.lsb_slider,1); lrow.addWidget(self.lsb_value,0)
        lwrap = QWidget(); lwrap.setLayout(lrow)
        form.addRow("Number of LSBs:", lwrap)
        form.addRow("Key:", self.key_edit)
        ctrl.setLayout(form)

        cap_box = QGroupBox("Capacity")
        self.cap_label = QLabel("Load cover + select ROI + add payload."); self.cap_label.setWordWrap(True)
        cap_v = QVBoxLayout(); cap_v.addWidget(self.cap_label); cap_box.setLayout(cap_v)

        key_box = QGroupBox("Final Key (copy for decoding)")
        key_h = QHBoxLayout()
        self.key_token_edit = QLineEdit(); self.key_token_edit.setReadOnly(True)
        self.copy_btn = QPushButton("Copy"); self.copy_btn.clicked.connect(self.copy_key_token)
        key_h.addWidget(self.key_token_edit); key_h.addWidget(self.copy_btn)
        key_box.setLayout(key_h)

        btns = QHBoxLayout()
        self.encode_btn = QPushButton("Derive Key & Encode (Placeholder)")
        self.encode_btn.clicked.connect(self.on_encode)
        btns.addWidget(self.encode_btn)

        left.addWidget(cov_box); left.addWidget(pay_box); left.addWidget(ctrl)
        left.addWidget(cap_box); left.addWidget(key_box); left.addLayout(btns); left.addStretch(1)

        right = QVBoxLayout()
        prev_box = QGroupBox("Image Preview (drag ROI)")
        pv = QVBoxLayout()
        self.cover_view = ImageView()
        self.cover_view.roiSelected.connect(self.set_roi_from_image)
        pv.addWidget(self.cover_view)
        prev_box.setLayout(pv)

        log_box = QGroupBox("Log")
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        lv = QVBoxLayout(); lv.addWidget(self.log_edit); log_box.setLayout(lv)
        right.addWidget(prev_box); right.addWidget(log_box)

        splitter = QSplitter()
        lw = QWidget(); lw.setLayout(left)
        rw = QWidget(); rw.setLayout(right)
        splitter.addWidget(lw); splitter.addWidget(rw); splitter.setSizes([470, 730])

        main = QVBoxLayout(self); main.addWidget(splitter)

    def copy_key_token(self):
        QtGui.QGuiApplication.clipboard().setText(self.key_token_edit.text())
        QMessageBox.information(self, "Copied", "Final Key copied to clipboard.")

    def load_cover(self, path: str):
        try:
            ext = Path(path).suffix.lower()
            if ext not in IMAGE_EXTS:
                raise ValueError("Unsupported image format.")

            self.cover_path = path

            if ext == ".gif":
                # Animated preview via QMovie
                movie = QMovie(path)
                if not movie.isValid():
                    raise ValueError("Failed to load GIF (invalid or unsupported).")
                self.cover_qpix = None
                self.cover_view.setImage(movie)

                # Use PIL to inspect first frame for img_info/capacity/ROI
                im = Image.open(path)
                try:
                    n_frames = getattr(im, "n_frames", 1)
                except Exception:
                    n_frames = 1
                self.img_info = {"size": im.size, "mode": im.mode, "bands": len(im.getbands())}
                self.cover_info.setText(
                    f"Path: {path}\nFormat: {im.format}, Mode: {im.mode}, "
                    f"Size: {im.size[0]}x{im.size[1]}\nFrames: {n_frames} (animated)"
                )
                self.log("Loaded GIF (animated). Preview shows animation; ROI applies to current frame area.")
                self.log("Note: Encoding will produce a static PNG; animation is not preserved.") #remove after improvement made

            else:
                # Static preview via QPixmap
                qpix = QPixmap(path)
                if qpix.isNull():
                    raise ValueError("Failed to load image.")
                self.cover_qpix = qpix
                self.cover_view.setImage(qpix)

                im = Image.open(path)
                self.img_info = {"size": im.size, "mode": im.mode, "bands": len(im.getbands())}
                self.cover_info.setText(
                    f"Path: {path}\nFormat: {im.format}, Mode: {im.mode}, Size: {im.size[0]}x{im.size[1]}"
                )

            # Reset ROI/capacity UI
            self.roi_img = None
            self.cap_label.setText("Select ROI to evaluate capacity.")
            self.update_capacity_label()
            self.log(f"Loaded image: {path}")

        except Exception as e:
            self.error(str(e))

    def set_roi_from_image(self, x0: int, y0: int, w: int, h: int):
        self.roi_img = (x0, y0, w, h)
        self.update_capacity_label()
        self.log(f"ROI selected (image): {self.roi_img}")

    def current_lsb(self) -> int:
        return self.lsb_slider.value()

    def update_capacity_label(self):
        if not self.cover_path or not self.roi_img or not self.img_info:
            self.cap_label.setText("Select ROI and add payload to check fit."); return
        lsb = self.current_lsb()
        x0,y0,w,h = self.roi_img
        channels = self.img_info["bands"]
        capacity_bits = w*h*channels*lsb
        payload_bits = self.payload_panel.payload_bits()
        overhead_bits = 84*8  # header
        if payload_bits is None:
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\nHeader requires {overhead_bits} bits.\nAdd payload (Text or File).")
        else:
            need = overhead_bits + payload_bits
            ok = need <= capacity_bits
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\n"
                f"Header+Payload need: {need} bits "
                f"({human_bytes((need+7)//8)})\n"
                f"Result: {'OK' if ok else 'Too large'}"
            )

    def on_encode(self):
        if not self.cover_path:
            self.error("Load a cover image."); return
        if not self.roi_img:
            self.error("Select an image ROI."); return
        payload = self.payload_panel.payload_bytes()
        if not payload:
            self.error("Enter payload text or choose a payload file."); return
        key = self.key_edit.text().strip()
        if not key:
            self.error("Key is required."); return

        try:
            # Open cover and validate mode
            cover_pil = Image.open(self.cover_path)
            cover_conv, channels, mode = _ensure_8bit_mode(cover_pil)
            if cover_conv.size != cover_pil.size:
                # Converted from unsupported mode; warn but proceed
                self.log("Cover converted to RGB for embedding.")

            # ROI & capacity
            lsb = self.current_lsb()
            x0, y0, w, h = self.roi_img
            W, H = cover_conv.size
            if w <= 0 or h <= 0 or x0 < 0 or y0 < 0 or x0+w > W or y0+h > H:
                self.error("ROI is out of bounds."); return
            capacity_bits = w*h*channels*lsb

            # KDF & header
            cover_id = cover_fingerprint(self.cover_path)
            full_salt = canonical_salt(lsb, self.roi_img, cover_id, "image")
            salt16 = full_salt[:16]
            kd = kdf_from_key(key, salt16)
            K_perm, K_bit, K_crypto, K_check, nonce = kd["K_perm"], kd["K_bit"], kd["K_crypto"], kd["K_check"], kd["nonce"]

            # Encrypt payload with stdlib keystream (deterministic)
            ks = _hmac_keystream(K_crypto, nonce, len(payload))
            cipher = _xor(payload, ks)

            header = build_header(1, lsb, self.roi_img, len(payload), cover_id, salt16, nonce, K_check)
            total_bits = (len(header) + len(cipher)) * 8
            if total_bits > capacity_bits:
                self.error(f"Payload too large for ROI capacity.\n"
                        f"Need {total_bits} bits, capacity {capacity_bits} bits."); return

            # BIT STREAM: header first, then cipher
            bitstream = []
            for b in header + cipher:
                for k in range(8):
                    bitstream.append((b >> (7-k)) & 1)
            n_bits = len(bitstream)

            # Prepare image bytes
            buf, imgW, imgH, ch = _img_bytes_and_geometry(cover_conv)

            # Slot model: slot = a pixel-channel (no plane). Planes are assigned per-bit.
            slots = w*h*channels
            perm = _make_perm(slots, K_perm)
            rot = K_bit[0] % lsb

            # Embed
            i_bit = 0
            while i_bit < n_bits:
                slot_index = i_bit // lsb
                plane = ((i_bit % lsb) + rot) % lsb
                if slot_index >= slots:
                    # Should not happen due to capacity check
                    self.error("Internal error: slot overflow during embed."); return
                # Map slot → (x,y,c)
                s = perm[slot_index]
                px = s // channels
                c  = s % channels
                rel_x = px % w
                rel_y = px // w
                x = x0 + rel_x
                y = y0 + rel_y
                idx = _pixel_byte_index(x, y, c, imgW, ch)

                # Set bit
                bit = bitstream[i_bit]
                buf[idx] = _set_bit(buf[idx], plane, bit)
                i_bit += 1

            # Write stego as PNG (lossless). Warn if source was JPEG.
            out_path = str(Path(self.cover_path).with_suffix("")) + "_stego.png"
            Image.frombytes(mode, (imgW, imgH), bytes(buf)).save(out_path, format="PNG")
            
            # NEW: write sidecar meta with absolute cover path for auto-load during decode
            try:
                meta_path = Path(out_path).with_suffix(".meta")
                meta_path.write_text(str(Path(self.cover_path).absolute()), encoding="utf-8")
                self.log(f"Saved meta: {meta_path}")
            except Exception as _e:
                self.log(f"[WARN] Could not write meta file: {_e}")

            token = make_key_token("image", lsb, self.roi_img, salt16, K_check)
            self.key_token_edit.setText(token)

            self.log(f"Derived token: {token}")
            self.log(f"Header bytes={len(header)} salt16={salt16.hex()} bit_rot={rot} order_seed={int.from_bytes(K_perm,'little')}")
            if Path(self.cover_path).suffix.lower() in {".jpg", ".jpeg"}:
                self.log("Note: Source was JPEG. Bits are saved losslessly to PNG as '..._stego.png'. "
                        "Re-saving as JPEG will corrupt hidden data.")
            QMessageBox.information(self, "Encode complete",
                f"Embedded {len(payload)} bytes (encrypted) with a {len(header)}-byte header.\n"
                f"Saved: {out_path}\nCopy the Final Key for decoding.")
        except Exception as e:
            self.error(str(e))


    def log(self, msg: str):
        self.log_edit.append(msg)

    def error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.log(f"[ERROR] {msg}")


# ---------- IMAGE DECODE ----------
class ImageDecodeTab(QWidget):
    def __init__(self):
        super().__init__()
        self.stego_path: str | None = None

        root = QVBoxLayout(self)
        media_box = QGroupBox("Stego Image")
        mv = QVBoxLayout()
        self.stego_drop = DropLabel("a stego image (.bmp/.png/.gif/.jpg/.jpeg)", IMAGE_EXTS)
        self.stego_drop.fileDropped.connect(self.load_stego)
        self.media_info = QLabel("No stego image loaded"); self.media_info.setWordWrap(True)
        mv.addWidget(self.stego_drop); mv.addWidget(self.media_info)
        media_box.setLayout(mv)

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

    def load_stego(self, path: str):
        try:
            ext = Path(path).suffix.lower()
            if ext not in IMAGE_EXTS: raise ValueError("Unsupported image format.")
            self.stego_path = path
            im = Image.open(path)
            self.media_info.setText(f"Path: {path}\nFormat: {im.format}, Mode: {im.mode}, Size: {im.size[0]}x{im.size[1]}")
            self.log(f"Loaded stego image: {path}")
        except Exception as e:
            self.error(str(e))

    def on_inspect(self):
        if not self.stego_path:
            self.error("Load a stego image first."); return
        token = self.key_token_edit.text().strip()
        user_key = self.user_key_edit.text().strip()
        if not token or not user_key:
            self.error("Provide both Final Key token and the user key."); return
        try:
            info = parse_key_token(token)
            if info["media_kind"] != "image":
                self.error("Final Key is not for an image."); return

            # Prepare image
            im = Image.open(self.stego_path)
            conv, channels, mode = _ensure_8bit_mode(im)
            W, H = conv.size
            x0, y0, w, h = info["roi"]
            if w <= 0 or h <= 0 or x0 < 0 or y0 < 0 or x0+w > W or y0+h > H:
                self.error("Token ROI is out of bounds for this image."); return

            # Derive keys and reconstruct bitstream just for header (84 bytes)
            kd = kdf_from_key(user_key, info["salt16"])
            K_perm, K_bit, K_crypto, K_check, nonce = kd["K_perm"], kd["K_bit"], kd["K_crypto"], kd["K_check"], kd["nonce"]

            # Quick key check
            if K_check != info["kcheck4"]:
                self.error("Wrong user key for this Final Key (K_check mismatch)."); return

            # Read header bits
            target_bits = 84 * 8
            buf, imgW, imgH, ch = _img_bytes_and_geometry(conv)
            lsb = info["lsb"]; rot = K_bit[0] % lsb
            slots = w*h*channels
            if target_bits > slots * lsb:
                self.error("Stego ROI cannot even hold a header; file/token mismatch."); return
            perm = _make_perm(slots, K_perm)

            bits = []
            for i_bit in range(target_bits):
                slot_index = i_bit // lsb
                plane = ((i_bit % lsb) + rot) % lsb
                s = perm[slot_index]
                px = s // channels
                c  = s % channels
                rel_x = px % w
                rel_y = px // w
                x = x0 + rel_x
                y = y0 + rel_y
                idx = _pixel_byte_index(x, y, c, imgW, ch)
                bits.append(_get_bit(buf[idx], plane))
            # Bits → bytes
            b = bytearray()
            for i in range(0, len(bits), 8):
                byte = 0
                for k in range(8):
                    byte = (byte << 1) | bits[i+k]
                b.append(byte)

            hdr = parse_header(bytes(b))

            # Sanity checks: lsb/roi must match token
            if hdr["lsb"] != lsb or hdr["roi"] != tuple(info["roi"]):
                self.error("Header mismatch (ROI/LSB do not match Final Key)."); return

            self.log(f"Header OK. version={hdr['version']} payload_len={hdr['payload_len']}")
            self.log(f"salt16={hdr['salt16'].hex()} nonce12={hdr['nonce12'].hex()} cover_fp16={hdr['cover_fp16'].hex()}")
            QMessageBox.information(self, "Inspect",
                f"Header read OK.\nPayload length: {hdr['payload_len']} bytes.\nReady to decode.")
        except Exception as e:
            self.error(str(e))

    def on_decode(self):
        if not self.stego_path:
            self.error("Load a stego image first."); return
        token = self.key_token_edit.text().strip()
        user_key = self.user_key_edit.text().strip()
        if not token or not user_key:
            self.error("Provide both Final Key token and the user key."); return

        try:
            info = parse_key_token(token)
            if info["media_kind"] != "image":
                self.error("Final Key is not for an image."); return

            im = Image.open(self.stego_path)
            conv, channels, mode = _ensure_8bit_mode(im)
            W, H = conv.size
            x0, y0, w, h = info["roi"]
            if w <= 0 or h <= 0 or x0 < 0 or y0 < 0 or x0+w > W or y0+h > H:
                self.error("Token ROI is out of bounds for this image."); return

            kd = kdf_from_key(user_key, info["salt16"])
            K_perm, K_bit, K_crypto, K_check, _nonce_from_kdf = (
                kd["K_perm"], kd["K_bit"], kd["K_crypto"], kd["K_check"], kd["nonce"]
            )
            if K_check != info["kcheck4"]:
                self.error("Wrong user key for this Final Key (K_check mismatch)."); return

            buf, imgW, imgH, ch = _img_bytes_and_geometry(conv)
            lsb = info["lsb"]; rot = K_bit[0] % lsb
            slots = w*h*channels
            perm = _make_perm(slots, K_perm)

            # ---- helper that supports a starting bit offset ----
            def _read_bits(n_bits: int, start_bit: int = 0) -> bytes:
                if n_bits + start_bit > slots * lsb:
                    raise ValueError("Requested bits exceed ROI capacity.")
                bits = []
                for i_bit in range(start_bit, start_bit + n_bits):
                    slot_index = i_bit // lsb
                    plane = ((i_bit % lsb) + rot) % lsb
                    s = perm[slot_index]
                    px = s // channels
                    c  = s % channels
                    rel_x = px % w
                    rel_y = px // w
                    x = x0 + rel_x
                    y = y0 + rel_y
                    idx = _pixel_byte_index(x, y, c, imgW, ch)
                    bits.append(_get_bit(buf[idx], plane))
                out = bytearray()
                for i in range(0, len(bits), 8):
                    byte = 0
                    for k in range(8):
                        byte = (byte << 1) | bits[i+k]
                    out.append(byte)
                return bytes(out)

            # read + verify header (84 bytes = 672 bits) starting at bit 0
            HEADER_BYTES = 84
            header_bytes = _read_bits(HEADER_BYTES * 8, start_bit=0)
            hdr = parse_header(header_bytes)
            if hdr["lsb"] != lsb or hdr["roi"] != tuple(info["roi"]):
                self.error("Header mismatch (ROI/LSB do not match Final Key)."); return
            if hdr["salt16"] != info["salt16"]:
                self.error("Header salt does not match Final Key salt."); return

            payload_len = hdr["payload_len"]
            total_bits_needed = (HEADER_BYTES + payload_len) * 8
            if total_bits_needed > slots * lsb:
                self.error("Stego does not contain the declared payload length (capacity shortfall)."); return

            # read payload bits starting AFTER the header
            payload_start = HEADER_BYTES * 8
            cipher_bytes = _read_bits(payload_len * 8, start_bit=payload_start)

            # decrypt via stdlib keystream (nonce from header is authoritative)
            ks = _hmac_keystream(K_crypto, hdr["nonce12"], len(cipher_bytes))
            payload = _xor(cipher_bytes, ks)

            out_dir = Path(self.stego_path).parent
            out_base = Path(self.stego_path).stem

            ext, label, is_text = guess_image_or_text_ext(payload)
            dest = out_dir / f"{out_base}_payload{ext}"

            if is_text and ext == ".txt":
                # Save as text
                txt = payload.decode("utf-8", errors="strict")
                with open(dest, "w", encoding="utf-8") as f:
                    f.write(txt)
                # preview = txt if len(txt) < 500 else txt[:500] + "…"
                # self.log("Decoded UTF-8 preview:\n" + preview)
            else:
                # Save raw bytes (images or unknown binary)
                with open(dest, "wb") as f:
                    f.write(payload)

            self.log(f"Saved payload as {dest} ({label}, {human_bytes(len(payload))})")
            QMessageBox.information(
                self, "Decode complete",
                f"Recovered {len(payload)} bytes.\nDetected: {label}\nSaved to:\n{dest}"
            )
            
            # --- Try to auto-load the original cover for side-by-side preview ---
            cover_guess = None
            try:
                # 1) Prefer sidecar .meta created during encode (absolute cover path)
                meta = Path(self.stego_path).with_suffix(".meta")
                if meta.exists():
                    candidate = meta.read_text(encoding="utf-8").strip()
                    if candidate and Path(candidate).exists():
                        # Verify the meta cover matches the header fingerprint
                        try:
                            if cover_fingerprint(candidate) == hdr["cover_fp16"]:
                                cover_guess = candidate
                                self.log(f"Auto-loaded cover from meta: {cover_guess}")
                            else:
                                self.log("[WARN] Meta cover fingerprint mismatch; not auto-loading.")
                        except Exception as _ve:
                            self.log(f"[WARN] Could not verify meta cover fingerprint: {_ve}")
                else:
                    # 2) Heuristic: if stego looks like '<name>_stego.png', try '<name>.(png/jpg/...)'
                    stem = Path(self.stego_path).stem
                    if stem.endswith("_stego"):
                        base = stem[:-6]
                        for ext_guess in [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff"]:
                            guess = Path(self.stego_path).with_name(base + ext_guess)
                            if guess.exists():
                                try:
                                    if cover_fingerprint(guess) == hdr["cover_fp16"]:
                                        cover_guess = str(guess)
                                        self.log(f"Guessed and verified cover: {cover_guess}")
                                        break
                                except Exception:
                                    pass
            except Exception as _e:
                self.log(f"[WARN] Auto-cover lookup failed: {_e}")
            
            # Launch preview dialog
            dlg = DecodePreviewDialog(
                stego_path=self.stego_path,
                payload_path=str(dest),
                payload_label=label,
                is_text=is_text,
                cover_path=cover_guess  # <-- NEW
            )
            dlg.exec()
        except Exception as e:
            self.error(str(e))

    def log(self, msg: str):
        self.log_edit.append(msg)

    def error(self, msg: str):
        QMessageBox.critical(self, "Error", msg)
        self.log(f"[ERROR] {msg}")


# ---------- Suite ----------
class ImageSuite(QWidget):
    def __init__(self):
        super().__init__()
        tabs = QTabWidget()
        tabs.addTab(ImageEncodeTab(), "Encode")
        tabs.addTab(ImageDecodeTab(), "Decode")
        lay = QVBoxLayout(self); lay.addWidget(tabs)
