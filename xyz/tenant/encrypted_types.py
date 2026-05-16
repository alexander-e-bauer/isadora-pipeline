"""Fernet-encrypted SQLAlchemy column types — mirrored from server.

Engine reads the same ENCRYPTION_KEY env var that server uses.  The
module-level _fernet_cache is intentionally separate from server's cache
(different process), but the Fernet key and all encryption/decryption
logic are byte-for-byte identical so ciphertext written by server is
readable by engine and vice-versa.

Tests may inject a known Fernet instance by setting:
    import xyz.tenant.encrypted_types as et
    et._fernet_cache = Fernet(some_key)
"""
from __future__ import annotations

import base64
import math
import os
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy import String, Text
from sqlalchemy.types import TypeDecorator


class EncryptedColumnError(ValueError):
    """Raised when an encrypted column value cannot be decrypted."""


# Module-level cache — reset to None in tests to force reconstruction.
_fernet_cache: Fernet | None = None


def _get_fernet() -> Fernet:
    """Return a cached Fernet instance built from the ENCRYPTION_KEY env var."""
    global _fernet_cache
    if _fernet_cache is None:
        key = os.environ.get("ENCRYPTION_KEY", "")
        if not key:
            raise RuntimeError(
                "ENCRYPTION_KEY env var is not set — "
                "engine cannot decrypt tenant PII columns."
            )
        _fernet_cache = Fernet(key.encode())
    return _fernet_cache


def _ciphertext_length_for(plaintext_length: int) -> int:
    """Conservative upper-bound VARCHAR length for ciphertext.

    Matches server's implementation exactly so column sizes are identical.
    """
    padded = plaintext_length + 32
    raw_bytes = 57 + math.ceil(padded / 16) * 16
    b64_len = math.ceil(raw_bytes / 3) * 4
    return b64_len


class EncryptedText(TypeDecorator):
    """Fernet-encrypted TEXT column — mirrors server's EncryptedText."""

    impl = Text
    cache_ok = True

    _fernet: Fernet | None = None

    def _fernet_instance(self) -> Fernet:
        if self._fernet is not None:
            return self._fernet
        return _get_fernet()

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        plaintext = value if isinstance(value, bytes) else str(value).encode()
        token = self._fernet_instance().encrypt(plaintext)
        return token.decode()

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        token = value if isinstance(value, bytes) else value.encode()
        try:
            plaintext_bytes = self._fernet_instance().decrypt(token)
        except InvalidToken as exc:
            raise EncryptedColumnError(
                "Failed to decrypt EncryptedText column value — "
                "key mismatch or ciphertext corruption."
            ) from exc
        return plaintext_bytes.decode()


class EncryptedString(TypeDecorator):
    """Fernet-encrypted VARCHAR column — mirrors server's EncryptedString."""

    impl = String
    cache_ok = True

    _fernet: Fernet | None = None

    def __init__(self, length: int, *args: Any, **kwargs: Any) -> None:
        cipher_length = _ciphertext_length_for(length)
        super().__init__(cipher_length, *args, **kwargs)
        self._plaintext_max = length

    def _fernet_instance(self) -> Fernet:
        if self._fernet is not None:
            return self._fernet
        return _get_fernet()

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        plaintext = value if isinstance(value, bytes) else str(value).encode()
        token = self._fernet_instance().encrypt(plaintext)
        return token.decode()

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        token = value if isinstance(value, bytes) else value.encode()
        try:
            plaintext_bytes = self._fernet_instance().decrypt(token)
        except InvalidToken as exc:
            raise EncryptedColumnError(
                "Failed to decrypt EncryptedString column value — "
                "key mismatch or ciphertext corruption."
            ) from exc
        return plaintext_bytes.decode()
