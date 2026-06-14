"""
logos/config.py — Provider configuration for LOGOS
====================================================

Manages ~/.logos/config.json

Supports:
  openai      → OpenAI API  (GPT-4o, GPT-4 Turbo, ...)
  anthropic   → Claude API  (Claude 3.5 Sonnet, ...)
  gemini      → Google API  (Gemini 1.5 Pro, ...)
  azure       → Azure OpenAI (your deployment)
  ollama      → Local Ollama (LLaMA, Qwen, Mistral, ...)
  foundry     → Azure AI Foundry 6-agent pipeline (advanced)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

LOGOS_HOME = Path.home() / ".logos"
CONFIG_FILE = LOGOS_HOME / "config.json"

PROVIDERS: dict[str, dict[str, Any]] = {
    "openai": {
        "label":         "OpenAI",
        "base_url":      "https://api.openai.com/v1",
        "models":        ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
        "default_model": "gpt-4o",
        "key_hint":      "sk-...",
    },
    "anthropic": {
        "label":         "Anthropic (Claude)",
        "base_url":      "https://api.anthropic.com/v1",
        "models":        ["claude-opus-4-5", "claude-sonnet-4-5", "claude-3-5-haiku-20241022"],
        "default_model": "claude-opus-4-5",
        "key_hint":      "sk-ant-...",
    },
    "gemini": {
        "label":         "Google Gemini",
        "base_url":      "https://generativelanguage.googleapis.com/v1beta/openai",
        "models":        ["gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash"],
        "default_model": "gemini-2.0-flash",
        "key_hint":      "AIza...",
    },
    "azure": {
        "label":         "Azure OpenAI",
        "base_url":      None,           # user provides endpoint
        "models":        ["gpt-4o", "gpt-4", "gpt-35-turbo"],
        "default_model": "gpt-4o",
        "key_hint":      "Your Azure API key",
        "needs_endpoint": True,
    },
    "ollama": {
        "label":         "Local Ollama (no API key needed)",
        "base_url":      "http://localhost:11434/v1",
        "models":        ["llama3.2", "qwen2.5", "mistral", "deepseek-r1"],
        "default_model": "llama3.2",
        "api_key":       "ollama",       # dummy — Ollama ignores it
    },
    "foundry": {
        "label":         "Azure AI Foundry  (6-agent pipeline + web search)",
        "base_url":      None,
        "models":        [],
        "default_model": "",
        "needs_endpoint": True,
        "advanced":      True,
    },
}


class Config:
    """
    Thin wrapper around ~/.logos/config.json.

    Usage
    -----
        cfg = Config()
        cfg.provider   # → "openai"
        cfg.api_key    # → "sk-..."
        cfg.model      # → "gpt-4o"
        cfg.set("model", "gpt-4o-mini")
        cfg.save()
    """

    def __init__(self) -> None:
        LOGOS_HOME.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, Any] = {}
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if CONFIG_FILE.exists():
            try:
                self._data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def save(self) -> None:
        CONFIG_FILE.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def is_configured(self) -> bool:
        provider = self._data.get("provider", "")
        if not provider:
            return False
        if provider == "ollama":
            return True          # no key needed
        return bool(self._data.get("api_key"))

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def provider(self) -> str:
        return self._data.get("provider", "openai")

    @property
    def api_key(self) -> str:
        prov = PROVIDERS.get(self.provider, {})
        return self._data.get("api_key") or prov.get("api_key", "")

    @property
    def model(self) -> str:
        return self._data.get("model", "gpt-4o")

    @property
    def base_url(self) -> str | None:
        custom = self._data.get("base_url")
        if custom:
            return custom
        return PROVIDERS.get(self.provider, {}).get("base_url")

    @property
    def azure_endpoint(self) -> str | None:
        return self._data.get("azure_endpoint")

    @property
    def azure_project_endpoint(self) -> str | None:
        return self._data.get("azure_project_endpoint")

    @property
    def provider_label(self) -> str:
        return PROVIDERS.get(self.provider, {}).get("label", self.provider)

    # ── OpenAI client factory ─────────────────────────────────────────────────

    def build_openai_client(self):
        """Returns an openai.OpenAI or openai.AzureOpenAI client."""
        from openai import OpenAI, AzureOpenAI

        if self.provider == "azure":
            return AzureOpenAI(
                azure_endpoint=self.azure_endpoint or "",
                api_key=self.api_key,
                api_version="2024-08-01-preview",
            )
        elif self.provider == "foundry":
            import os
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
            api_key  = os.getenv("AZURE_OPENAI_API_KEY", "")
            if endpoint and api_key:
                return AzureOpenAI(
                    azure_endpoint=endpoint,
                    api_key=api_key,
                    api_version="2024-08-01-preview",
                )

        kwargs: dict[str, Any] = {"api_key": self.api_key or "sk-no-key"}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)
