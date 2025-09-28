# INF2005_P1_Steganography_Project_Team_2




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