"""AES-128-CTR encryption for decoded pixel data."""

from secrets import token_bytes

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes


def encrypt_ctr(pixel_data: bytes, key: bytes) -> tuple[bytes, bytes]:
    """Return AES-128-CTR ciphertext and its initial counter block separately."""
    if len(key) != 16:
        raise ValueError("AES-128 requires a 16-byte key")

    initial_counter = token_bytes(algorithms.AES.block_size // 8)
    encryptor = Cipher(
        algorithms.AES(key), modes.CTR(initial_counter)
    ).encryptor()
    ciphertext = encryptor.update(pixel_data) + encryptor.finalize()
    return ciphertext, initial_counter
