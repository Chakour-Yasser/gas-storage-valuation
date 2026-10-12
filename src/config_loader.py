"""Load the project configuration from config/config.toml."""

import tomllib
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config.toml"


def parse_date(text):
    """Parse an ISO date string 'YYYY-MM-DD' into a datetime.date."""
    return date.fromisoformat(text)


def load_config(path=None):
    """Read the TOML config file and return it as a nested dict."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    with open(path, "rb") as file:
        config = tomllib.load(file)
    return config
