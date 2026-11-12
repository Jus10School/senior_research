"""AES-128-CFB encryption for decoded pixel data."""

from secrets import token_bytes

from cryptography.hazmat.decrepit.ciphers.modes import CFB
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


def encrypt_cfb(pixel_data: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Return the IV and AES-128-CFB ciphertext separately."""
    if len(key) != 16:
        raise ValueError("AES-128 requires a 16-byte key")

    iv = token_bytes(algorithms.AES.block_size // 8)
    encryptor = Cipher(algorithms.AES(key), CFB(iv)).encryptor()
    ciphertext = encryptor.update(pixel_data) + encryptor.finalize()
    return iv, ciphertext
