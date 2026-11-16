"""Data ingestion, dynamic table registration, and schema profiling."""

from src.agent.data.ingestion import SessionDatabase
from src.agent.data.profiling import TableProfile, format_profile_for_prompt, profile_table

__all__ = [
    "SessionDatabase",
    "TableProfile",
    "format_profile_for_prompt",
    "profile_table",
]
