"""Environment variable helper utilities."""

import os


def getenv_bool(env_key: str, default: bool) -> bool:
    """Return one environment variable parsed as a boolean."""
    value = os.getenv(env_key)
    if value is None:
        return bool(default)
    value_lc = value.strip().lower()
    if value_lc in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value_lc in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean for {env_key}: {value!r}")


def getenv_int(env_key: str, default: int) -> int:
    """Return one environment variable parsed as an integer."""
    value = os.getenv(env_key)
    if value is None:
        return int(default)
    return int(value)


def getenv_float(env_key: str, default: float) -> float:
    """Return one environment variable parsed as a float."""
    value = os.getenv(env_key)
    if value is None:
        return float(default)
    return float(value)
