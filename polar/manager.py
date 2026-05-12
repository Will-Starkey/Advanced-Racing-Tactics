"""
PolarManager — upload, store, retrieve, and activate polar files.
Polars are persisted as JSON in data/polars/.
"""

import json
import os
from dataclasses import asdict
from typing import Optional

from .models import PolarData
from .parser import PolarParser

POLAR_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "polars")
os.makedirs(POLAR_DIR, exist_ok=True)


class PolarManager:
    def __init__(self):
        self._active: Optional[PolarData] = None

    @property
    def active(self) -> Optional[PolarData]:
        return self._active

    def upload(self, filename: str, content: str, boat_name: str) -> PolarData:
        """Parse and persist a polar file.  Returns the PolarData."""
        polar = PolarParser.parse(filename, content, boat_name)
        self._save(polar)
        return polar

    def set_active(self, boat_name: str) -> PolarData:
        """Load a stored polar and mark it as active."""
        polar = self._load(boat_name)
        self._active = polar
        return polar

    def list_boats(self) -> list:
        return [
            f.replace(".json", "")
            for f in os.listdir(POLAR_DIR)
            if f.endswith(".json")
        ]

    # ── Persistence ───────────────────────────────────────────────

    def _save(self, polar: PolarData):
        path = self._path(polar.boat_name)
        with open(path, "w") as f:
            json.dump(asdict(polar), f, indent=2)

    def _load(self, boat_name: str) -> PolarData:
        path = self._path(boat_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No polar stored for '{boat_name}'")

        with open(path) as f:
            data = json.load(f)

        polar = PolarData(
            boat_name=data["boat_name"],
            source_file=data["source_file"],
            tws_values=data["tws_values"],
            twa_values=data["twa_values"],
        )

        # JSON serialises dict keys as strings — restore float keys
        polar.bsp_matrix = {
            float(twa): {float(tws): bsp for tws, bsp in row.items()}
            for twa, row in data["bsp_matrix"].items()
        }
        polar.upwind_vmg_angles = {
            float(k): v for k, v in data["upwind_vmg_angles"].items()
        }
        polar.downwind_vmg_angles = {
            float(k): v for k, v in data["downwind_vmg_angles"].items()
        }
        return polar

    def _path(self, boat_name: str) -> str:
        safe = boat_name.lower().replace(" ", "_").replace("/", "_")
        return os.path.join(POLAR_DIR, f"{safe}.json")
