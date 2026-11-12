"""AES-128-ECB encryption for decoded pixel data."""

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def encrypt_ecb(pixel_data: bytes, key: bytes) -> bytes:
    """Return AES-128-ECB ciphertext for decoded pixel data."""
    if len(key) != 16:
        raise ValueError("AES-128 requires a 16-byte key")

    padder = padding.PKCS7(algorithms.AES.block_size).padder()
    padded_data = padder.update(pixel_data) + padder.finalize()

    encryptor = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    return encryptor.update(padded_data) + encryptor.finalize()