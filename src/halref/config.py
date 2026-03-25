"""Configuration management for halref."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ExtractionConfig(BaseModel):
    text_extractors: list[str] = Field(default=["pdfminer"])
    field_parsers: list[str] = Field(default=["regex", "heuristic"])
    ref_pages: str = ""

    def page_range(self) -> tuple[int, int] | None:
        """Parse ref_pages string (1-indexed) into 0-indexed tuple."""
        if not self.ref_pages:
            return None
        parts = self.ref_pages.strip().split("-")
        if len(parts) == 1:
            p = int(parts[0]) - 1
            return (p, p + 1)
        start = int(parts[0]) - 1
        end = int(parts[1])  # 1-indexed end becomes exclusive 0-indexed
        return (start, end)


class APIConfig(BaseModel):
    enabled: bool = True
    api_key: str = ""
    mailto: str = ""


class MatchingWeights(BaseModel):
    title: float = 0.30
    authors: float = 0.25
    author_order: float = 0.15
    year: float = 0.15
    consensus: float = 0.15


class MatchingConfig(BaseModel):
    title_threshold: float = 0.85
    author_threshold: float = 0.6
    weights: MatchingWeights = Field(default_factory=MatchingWeights)


class AgentConfig(BaseModel):
    enable_retry: bool = True
    max_retries: int = 2


class LLMConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:8000/v1"
    model: str = ""
    api_key: str = ""


class Config(BaseSettings):
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    apis: dict[str, APIConfig] = Field(default_factory=lambda: {
        "semantic_scholar": APIConfig(),
        "crossref": APIConfig(),
        "dblp": APIConfig(),
        "openalex": APIConfig(),
        "acl_anthology": APIConfig(),
    })
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    model_config = {"env_prefix": "HALREF_"}

    @classmethod
    def from_file(cls, path: str | Path) -> Config:
        """Load config from a TOML file, with env var overrides."""
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p, "rb") as f:
            data = tomllib.load(f)
        return cls(**_flatten_apis(data))

    @classmethod
    def default(cls) -> Config:
        return cls()

    def get_api_config(self, name: str) -> APIConfig:
        return self.apis.get(name, APIConfig(enabled=False))

    def enabled_apis(self) -> list[str]:
        return [name for name, cfg in self.apis.items() if cfg.enabled]


def _flatten_apis(data: dict[str, Any]) -> dict[str, Any]:
    """Convert apis.semantic_scholar.api_key TOML structure to nested dict."""
    if "apis" in data and isinstance(data["apis"], dict):
        apis = {}
        for name, cfg in data["apis"].items():
            if isinstance(cfg, dict):
                apis[name] = APIConfig(**cfg)
            else:
                apis[name] = cfg
        data["apis"] = apis
    return data


def load_config(config_path: str | None = None) -> Config:
    """Load config from file or defaults, applying env var overrides.

    Loads .env file if present (does not override existing env vars).
    """
    import os

    # Load .env file if present
    _load_dotenv()

    if config_path:
        config = Config.from_file(config_path)
    else:
        # Check common locations
        for candidate in ["halref.toml", "config.toml", ".halref.toml"]:
            if Path(candidate).exists():
                config = Config.from_file(candidate)
                break
        else:
            config = Config.default()

    # Apply env var overrides for API keys
    env_keys = {
        "semantic_scholar": "SEMANTIC_SCHOLAR_API_KEY",
        "openalex": "OPENALEX_API_KEY",
        "crossref": "CROSSREF_MAILTO",
    }
    for api_name, env_var in env_keys.items():
        val = os.environ.get(env_var, "")
        if val and api_name in config.apis:
            if env_var.endswith("_MAILTO"):
                config.apis[api_name].mailto = val
            else:
                config.apis[api_name].api_key = val

    # LLM config from env
    for env_var, attr in [
        ("HALREF_LLM_BASE_URL", "base_url"),
        ("HALREF_LLM_MODEL", "model"),
        ("HALREF_LLM_API_KEY", "api_key"),
    ]:
        val = os.environ.get(env_var, "")
        if val:
            setattr(config.llm, attr, val)

    return config


def _load_dotenv() -> None:
    """Load .env file into os.environ (without overriding existing vars).

    Simple parser — no external dependency needed.
    Supports: KEY=value, KEY="value", KEY='value', and comments (#).
    """
    import os

    for candidate in [".env", "../.env"]:
        env_path = Path(candidate)
        if not env_path.exists():
            continue
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                # Remove surrounding quotes
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                # Don't override existing env vars
                if key not in os.environ or not os.environ[key]:
                    os.environ[key] = value
        break  # Only load first .env found
