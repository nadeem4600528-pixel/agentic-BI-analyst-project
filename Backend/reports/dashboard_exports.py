"""Dashboard export helpers."""
import json
from typing import Any, Mapping


def export_dashboard_json(dashboard: Mapping[str, Any]) -> str:
    return json.dumps(dashboard, indent=2, default=str)

