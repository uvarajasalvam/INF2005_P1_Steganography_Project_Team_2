# image_ui.py
import math
import os, io, zlib, hmac, hashlib, struct, base64
from pathlib import Path

from numpy.random import PCG64, Generator
import numpy as np
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

# ----- Image helpers -----
def _img_to_array(path: str):
    # Load with NO color conversion to avoid any LSB changes from color management.
    im = Image.open(path)
    if im.mode != "RGB":
        # Refuse non-RGB modes to keep exact byte fidelity of channels.
        raise ValueError(f"Image must be 24-bit RGB. Got mode={im.mode}. "
                         "Please use the PNG/BMP saved by this app (RGB).")
    return np.array(im, dtype=np.uint8), im


def _array_to_img(a: np.ndarray):
    return Image.fromarray(a, mode="RGB")

def _roi_view(a: np.ndarray, x0, y0, w, h) -> np.ndarray:
    return a[y0:y0+h, x0:x0+w, :]

def _bytes_from_array(roi_view: np.ndarray) -> np.ndarray:
    return roi_view.reshape(-1)  # 1-D uint8

def _bytes_to_bits(b: bytes) -> np.ndarray:
    arr = np.frombuffer(b, dtype=np.uint8)
    return np.unpackbits(arr)

def _bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size % 8 != 0:
        bits = np.pad(bits, (0, 8 - (bits.size % 8)))
    return bytes(np.packbits(bits.astype(np.uint8)))

def _write_bits_into_carriers(
    carriers: np.ndarray,
    bits: np.ndarray,
    lsb: int,
    bit_offset: int,
    positions: np.ndarray,
):
    """
    positions[i] points to the carrier (byte) that stores bits[i].
    We cycle bit-planes as (bit_offset + i) % lsb.
    All math is masked to 8 bits to avoid negative python-ints.
    """
    for i, (pos, b) in enumerate(zip(positions, bits)):
        plane = (bit_offset + i) % lsb
        mask = (1 << plane) & 0xFF                   # 8-bit mask
        v = int(carriers[pos]) & 0xFF                # read as int 0..255
        newv = (v & (~mask & 0xFF)) | ((int(b) << plane) & mask)
        carriers[pos] = np.uint8(newv)               # write back as uint8


def _read_bits_from_carriers(carriers: np.ndarray, n_bits: int, lsb: int, bit_offset: int, positions: np.ndarray) -> np.ndarray:
    out = np.zeros(n_bits, dtype=np.uint8)
    for i in range(n_bits):
        plane = (bit_offset + i) % lsb
        mask = 1 << plane
        out[i] = (carriers[positions[i]] & mask) >> plane
    return out

def _permute_positions(base_positions: np.ndarray, rng: Generator) -> np.ndarray:
    idx = base_positions.copy()
    rng.shuffle(idx)
    return idx

# ---------- Widgets ----------
class ImageView(QtWidgets.QFrame):
    roiSelected = Signal(int, int, int, int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self._pix = QPixmap()
        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._origin = None
        self._display_rect = QRect()
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
        w = self.width(); h = self.height()
        pw = self._pix.width(); ph = self._pix.height()
        scale = min(w/pw, h/ph)
        dw = int(pw*scale); dh = int(ph*scale)
        x0 = (w - dw)//2; y0 = (h - dh)//2
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
        if self._display_rect.width() == 0 or self._display_rect.height() == 0:
            return
        pw, ph = self._pix.width(), self._pix.height()
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
            if ext not in IMAGE_EXTS: raise ValueError("Unsupported image format.")
            self.cover_path = path
            qpix = QPixmap(path)
            if qpix.isNull(): raise ValueError("Failed to load image.")
            self.cover_qpix = qpix
            self.cover_view.setImage(qpix)
            im = Image.open(path)
            self.img_info = {"size": im.size, "mode": im.mode, "bands": len(im.getbands())}
            self.cover_info.setText(f"Path: {path}\nFormat: {im.format}, Mode: {im.mode}, Size: {im.size[0]}x{im.size[1]}")
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
        channels = 3
        capacity_bits = w*h*channels*lsb
        payload_bits = self.payload_panel.payload_bits()
        if payload_bits is None:
            self.cap_label.setText(f"ROI capacity ≈ {capacity_bits} bits\nAdd payload (Text or File).")
        else:
            ok = payload_bits <= capacity_bits
            self.cap_label.setText(
                f"ROI capacity ≈ {capacity_bits} bits\n"
                f"Payload: {payload_bits} bits ({human_bytes(payload_bits//8)})\n"
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
            # 0) Derive salts/keys and build header + Final Key token
            lsb = self.current_lsb()
            x0, y0, w, h = self.roi_img
            if w <= 0 or h <= 0:
                raise ValueError("ROI width/height must be positive.")
            cover_id = cover_fingerprint(self.cover_path)
            full_salt = canonical_salt(lsb, self.roi_img, cover_id, "image")
            salt16 = full_salt[:16]

            kd = kdf_from_key(key, salt16)
            K_perm, K_bit = kd["K_perm"], kd["K_bit"]
            K_crypto, K_check, nonce = kd["K_crypto"], kd["K_check"], kd["nonce"]  # crypto optional

            header = build_header(
                version=1,
                lsb=lsb,
                roi_xywh=self.roi_img,
                payload_len=len(payload),
                cover_fp16=cover_id,
                salt16=salt16,
                nonce12=nonce,
                kcheck4=K_check
            )
            token = make_key_token("image", lsb, self.roi_img, salt16, K_check)
            self.key_token_edit.setText(token)

            # 1) Load image and get ROI carriers (RGB uint8)
            arr, _ = _img_to_array(self.cover_path)        # RGB uint8
            H, W, C = arr.shape                            # C = 3
            if x0 + w > W or y0 + h > H:
                raise ValueError("ROI goes outside the image bounds.")
            roi_view = _roi_view(arr, x0, y0, w, h)[:, :, :3]

            # >>> Make ROI contiguous so writes persist, and flatten that.
            roi_buf = np.ascontiguousarray(roi_view)       # <<< key fix
            carriers = roi_buf.reshape(-1)                 # safe to write

            # 2) Prepare bits
            header_bits = _bytes_to_bits(header)           # 84 bytes -> 672 bits
            payload_bits = _bytes_to_bits(payload)
            total_bits = header_bits.size + payload_bits.size
            capacity_bits = carriers.size * lsb
            if total_bits > capacity_bits:
                raise ValueError(
                    f"Capacity too small: need {total_bits} bits, ROI provides {capacity_bits} bits (lsb={lsb})."
                )

            # 3) positions (packing)
            hdr_bits_n = header_bits.size
            pay_bits_n = payload_bits.size
            hdr_carriers = math.ceil(hdr_bits_n / lsb)
            pay_carriers = math.ceil(pay_bits_n / lsb)

            hdr_pos = (np.arange(hdr_bits_n, dtype=np.int64) // lsb)

            pay_carrier_base = np.arange(hdr_carriers, hdr_carriers + pay_carriers, dtype=np.int64)
            rng = rng_from_16_bytes(K_perm)
            perm_carriers = pay_carrier_base.copy()
            rng.shuffle(perm_carriers)
            pay_pos = perm_carriers[(np.arange(pay_bits_n, dtype=np.int64) // lsb)]

            # 4) plane offsets
            bit_offset_header = 0
            bit_offset_payload = int(K_bit[0] % lsb)

            # 5) write (into roi_buf)
            _write_bits_into_carriers(carriers, header_bits,  lsb, bit_offset_header, hdr_pos)
            _write_bits_into_carriers(carriers, payload_bits, lsb, bit_offset_payload, pay_pos)

            # >>> Write the modified ROI back into the image array before saving.
            arr[y0:y0+h, x0:x0+w, :3] = roi_buf            # <<< key fix

            # --- SELF-VERIFY HEADER right after writing (reads from modified roi_buf) ---
            try:
                hdr_bits_n = 84 * 8
                hdr_carriers = math.ceil(hdr_bits_n / lsb)
                hdr_pos_verify = (np.arange(hdr_bits_n, dtype=np.int64) // lsb)
                hdr_bits_back = _read_bits_from_carriers(roi_buf.reshape(-1), hdr_bits_n, lsb, 0, hdr_pos_verify)
                header_back = _bits_to_bytes(hdr_bits_back)
                self.log(f"[self-check] header bytes[:8] = {header_back[:8].hex()}")
                hchk = parse_header(header_back)
                self.log(f"[self-check] header OK. magic='STG1', payload_len={hchk['payload_len']}")
            except Exception as e:
                self.error(f"Self-verify failed BEFORE saving: {e}")
                return

            # 6) Save lossless (PNG/BMP)
            out = QFileDialog.getSaveFileName(
                self,
                "Save stego image",
                "",
                "PNG Image (*.png);;BMP Image (*.bmp)"
            )[0]
            if not out:
                return

            ext = os.path.splitext(out)[1].lower()
            img_to_save = Image.fromarray(arr, mode="RGB")
            if ext == ".png":
                # Plain PNG save (avoid optimize/gamma/icc surprises)
                try:
                    from PIL.PngImagePlugin import PngInfo
                    pnginfo = PngInfo()
                except Exception:
                    pnginfo = None
                img_to_save.save(out, format="PNG", optimize=False, compress_level=0,
                                icc_profile=None, pnginfo=pnginfo)
            else:
                img_to_save.save(out, format="BMP")

            self.log(f"Stego saved: {out}")
            QMessageBox.information(
                self, "Encode done",
                "Embedding complete. Final Key generated above; keep it safe for decoding."
            )
            self.log(f"Derived token: {token}")

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
        self.inspect_btn = QPushButton("Inspect Header (Placeholder)")
        self.decode_btn  = QPushButton("Decode Payload (Placeholder)")
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
        if not token:
            self.error("Paste the Final Key token first."); return
        try:
            info = parse_key_token(token)
            if info["media_kind"] != "image":
                raise ValueError("Final Key is not for an image.")
            # force ints (some parsers return numpy/int-like)
            lsb = int(info["lsb"])
            x0, y0, w, h = map(int, info["roi"])
            self.log(f"[token] kind=image lsb={lsb} roi=({x0},{y0},{w},{h}) salt16={info['salt16'].hex()}")

            # Load stego and make ROI carriers
            arr, _ = _img_to_array(self.stego_path)  # RGBA
            H, W, C = arr.shape
            if x0 < 0 or y0 < 0 or w <= 0 or h <= 0 or x0 + w > W or y0 + h > H:
                raise ValueError(f"ROI out of bounds for this image: image=({W}x{H}), roi=({x0},{y0},{w},{h})")
            roi_view = _roi_view(arr, x0, y0, w, h)[:, :, :3]   # keep only RGB
            carriers = roi_view.reshape(-1)                     # 1-D uint8

            # Read fixed 84-byte header (packed, offset=0)
            hdr_bits_n = 84 * 8
            hdr_carriers = math.ceil(hdr_bits_n / lsb)
            if hdr_carriers > carriers.size:
                raise ValueError(f"ROI too small to contain header: need {hdr_carriers} carriers, have {carriers.size}")
            hdr_pos = (np.arange(hdr_bits_n, dtype=np.int64) // lsb)
            hdr_bits = _read_bits_from_carriers(carriers, hdr_bits_n, lsb, 0, hdr_pos)
            header_bytes = _bits_to_bytes(hdr_bits)
            self.log(f"[inspect] header bytes[:8] = {header_bytes[:8].hex()}")

            try:
                hdr = parse_header(header_bytes)
            except Exception as e:
                # quick diagnose: brute-force lsb to see if token's lsb mismatched
                found = None
                for guess in range(1, 9):
                    nc = math.ceil(hdr_bits_n / guess)
                    if nc > carriers.size: 
                        continue
                    pos = (np.arange(hdr_bits_n, dtype=np.int64) // guess)
                    bits = _read_bits_from_carriers(carriers, hdr_bits_n, guess, 0, pos)
                    hb = _bits_to_bytes(bits)
                    if hb[:4] == b"STG1":
                        found = guess
                        break
                magic = header_bytes[:4]
                crc_calc = zlib.crc32(header_bytes[:80]) & 0xFFFFFFFF if len(header_bytes) >= 84 else None
                if found:
                    self.error(f"Header parse failed: {e}\n"
                            f"magic={magic!r} with lsb={lsb}; BUT looks like header present with lsb={found}. "
                            f"Use the matching token.")
                else:
                    self.error(f"Header parse failed: {e}\nmagic={magic!r}, len={len(header_bytes)}, crc_calc={crc_calc}")
                return

            # Show a quick summary
            QMessageBox.information(
                self, "Header",
                f"Version: {hdr['version']}\n"
                f"LSB: {hdr['lsb']}\n"
                f"ROI: {hdr['roi']}\n"
                f"Payload: {hdr['payload_len']} bytes"
            )
            self.log(f"Header OK. salt16={hdr['salt16'].hex()} kcheck4={hdr['kcheck4'].hex()}")

        except Exception as e:
            self.error(str(e))


    def on_decode(self):
        if not self.stego_path:
            self.error("Load a stego image first."); return
        token = self.key_token_edit.text().strip()
        user_key = self.user_key_edit.text().strip()
        if not token:
            self.error("Paste the Final Key token."); return
        if not user_key:
            self.error("Enter the User Key."); return

        try:
            info = parse_key_token(token)
            if info["media_kind"] != "image":
                raise ValueError("Final Key is not for an image.")
            lsb = int(info["lsb"])
            x0, y0, w, h = map(int, info["roi"])
            salt16 = info["salt16"]
            self.log(f"[token] kind=image lsb={lsb} roi=({x0},{y0},{w},{h}) salt16={salt16.hex()}")

            kd = kdf_from_key(user_key, salt16)
            K_perm, K_bit = kd["K_perm"], kd["K_bit"]
            bit_offset_payload = int(K_bit[0] % lsb)

            # Load stego image and get ROI carriers
            arr, _ = _img_to_array(self.stego_path)  # RGBA
            H, W, C = arr.shape
            if x0 < 0 or y0 < 0 or w <= 0 or h <= 0 or x0 + w > W or y0 + h > H:
                raise ValueError(f"ROI out of bounds: image=({W}x{H}), roi=({x0},{y0},{w},{h})")
            roi_view = _roi_view(arr, x0, y0, w, h)[:, :, :3]   # keep only RGB
            carriers = roi_view.reshape(-1)                     # 1-D uint8

            # --- Header (packed, offset=0, unpermuted) ---
            hdr_bits_n = 84 * 8
            hdr_carriers = math.ceil(hdr_bits_n / lsb)
            if hdr_carriers > carriers.size:
                raise ValueError(f"ROI too small to contain header: need {hdr_carriers} carriers, have {carriers.size}")
            hdr_pos = (np.arange(hdr_bits_n, dtype=np.int64) // lsb)
            hdr_bits = _read_bits_from_carriers(carriers, hdr_bits_n, lsb, 0, hdr_pos)
            header_bytes = _bits_to_bytes(hdr_bits)
            self.log(f"[decode] header[:8] = {header_bytes[:8].hex()}")
            hdr = parse_header(header_bytes)

            # --- Payload (permuted, packed, rotated) ---
            pay_len = hdr["payload_len"]
            pay_bits_n = pay_len * 8
            pay_carriers = math.ceil(pay_bits_n / lsb)
            if hdr_carriers + pay_carriers > carriers.size:
                raise ValueError(f"ROI too small for header+payload: need {hdr_carriers+pay_carriers}, have {carriers.size}")

            pay_carrier_base = np.arange(hdr_carriers, hdr_carriers + pay_carriers, dtype=np.int64)
            rng = rng_from_16_bytes(K_perm)
            perm_carriers = pay_carrier_base.copy()
            rng.shuffle(perm_carriers)

            pay_pos = perm_carriers[(np.arange(pay_bits_n, dtype=np.int64) // lsb)]
            pay_bits = _read_bits_from_carriers(carriers, pay_bits_n, lsb, bit_offset_payload, pay_pos)
            payload_bytes = _bits_to_bytes(pay_bits)[:pay_len]

            out = QFileDialog.getSaveFileName(self, "Save extracted payload", "", "All Files (*)")[0]
            if out:
                with open(out, "wb") as f:
                    f.write(payload_bytes)
                self.log(f"Payload saved: {out}")
                QMessageBox.information(self, "Decode done", "Payload extracted and saved.")
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
