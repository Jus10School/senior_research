"""AES-128-CBC encryption for decoded pixel data."""

from secrets import token_bytes

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def encrypt_cbc(pixel_data: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Return the IV and AES-128-CBC ciphertext for pixel data separately."""
    if len(key) != 16:
        raise ValueError("AES-128 requires a 16-byte key")

    iv = token_bytes(algorithms.AES.block_size // 8)
    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(pixel_data) + padder.finalize()

    encryptor = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
    ciphertext = encryptor.update(padded_data) + encryptor.finalize()
    return iv, ciphertext