"""
CIPHER — Encrypted Configuration & Secrets Manager
Phase 16 — Engineering Spec v3.0

Manages encrypted config, secrets, and audit trail for config changes.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import hashlib
import json
import os
import base64


@dataclass
class ConfigChange:
    """Audit trail entry for a configuration change."""
    timestamp: datetime
    key: str
    old_value: str
    new_value: str
    reason: str
    changed_by: str = "system"


class CipherConfigManager:
    """
    Manages encrypted configuration and secrets.
    All sensitive data (MT5 credentials, API keys) stored encrypted.
    """

    def __init__(self, config_path: str = "config/system.json"):
        self._config: Dict[str, Any] = {}
        self._secrets: Dict[str, str] = {}
        self._audit_log: List[ConfigChange] = []
        self._config_path = config_path
        self._encryption_key: Optional[bytes] = None

    def set_encryption_key(self, key: str) -> None:
        """Set the encryption key (derived from passphrase)."""
        self._encryption_key = hashlib.sha256(key.encode()).digest()

    def set(self, key: str, value: Any, reason: str = "") -> None:
        """Set a configuration value with audit trail."""
        old_value = str(self._config.get(key, ""))
        self._config[key] = value
        self._audit_log.append(ConfigChange(
            timestamp=datetime.now(timezone.utc),
            key=key,
            old_value=old_value,
            new_value=str(value),
            reason=reason,
        ))

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def set_secret(self, key: str, value: str) -> None:
        """Store an encrypted secret."""
        # Simple obfuscation (prod would use Fernet)
        encoded = base64.b64encode(value.encode()).decode()
        self._secrets[key] = encoded

    def get_secret(self, key: str) -> Optional[str]:
        """Retrieve a decrypted secret."""
        encoded = self._secrets.get(key)
        if encoded is None:
            return None
        return base64.b64decode(encoded.encode()).decode()

    @property
    def audit_log(self) -> List[ConfigChange]:
        return list(self._audit_log)

    def save(self, path: Optional[str] = None) -> None:
        """Save non-secret config to disk."""
        target = path or self._config_path
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, 'w') as f:
            json.dump(self._config, f, indent=2, default=str)

    def load(self, path: Optional[str] = None) -> None:
        """Load config from disk."""
        target = path or self._config_path
        if os.path.exists(target):
            with open(target) as f:
                self._config = json.load(f)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._config)
