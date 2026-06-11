"""
Centralized configuration for the task engine.

Loads settings from environment variables (or .env file) using pydantic-settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider ────────────────────────────────────────────────────
    openai_api_key: str = Field(description="OpenAI API key")
    openai_model_name: str = Field(default="gpt-4o", description="LLM model name")
    openai_temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="LLM sampling temperature"
    )

    # ── Pinecone Vector Database ────────────────────────────────────────
    pinecone_api_key: str = Field(description="Pinecone API key")
    pinecone_index_name: str = Field(
        default="taskengine-knowledge", description="Pinecone index name"
    )
    pinecone_cloud: str = Field(
        default="aws", description="Pinecone cloud provider (aws, gcp, azure)"
    )
    pinecone_region: str = Field(
        default="us-east-1", description="Pinecone region"
    )
    pinecone_dimension: int = Field(
        default=1536,
        description="Embedding dimension (1536 for text-embedding-3-small)",
    )
    pinecone_metric: str = Field(
        default="cosine", description="Distance metric for vector similarity"
    )

    # ── Engine Parameters ───────────────────────────────────────────────
    max_steps: int = Field(
        default=15, ge=1, le=50, description="Maximum steps per task plan"
    )
    max_replans: int = Field(
        default=2, ge=0, le=5, description="Maximum re-planning attempts"
    )
    step_timeout_seconds: int = Field(
        default=120, ge=10, le=600, description="Timeout per step execution"
    )

    # ── API Server ──────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")

    # ── Workspace ───────────────────────────────────────────────────────
    workspace_dir: str = Field(
        default="./workspace",
        description="Directory for file operations (sandboxed)",
    )

    @property
    def workspace_path(self) -> Path:
        """Resolved absolute path to the workspace directory."""
        path = Path(self.workspace_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path


def get_settings() -> Settings:
    """Create and return a Settings instance (cached per call-site if needed)."""
    return Settings()  # type: ignore[call-arg]
