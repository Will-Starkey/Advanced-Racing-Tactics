"""
Polar file parser — supports three formats:

  1. Expedition (.pol / .txt)
       Twa/Tws  6    8   10   12   16   20
       52       5.4  6.3  6.8  7.1  7.4  7.5
       ...

  2. Simple CSV
       TWA,6,8,10,12,16,20
       52,5.4,6.3,6.8,7.1,7.4,7.5
       ...

  3. ORC JSON
       {"speeds":[6,8,10],"angles":[52,60,75],"values":[[bsp,...],...]}
"""

import csv
import io
import json

from .models import PolarData


class PolarParser:

    @classmethod
    def parse(cls, filename: str, content: str, boat_name: str) -> PolarData:
        ext = filename.rsplit(".", 1)[-1].lower()
        if ext in ("pol", "txt"):
            return cls._parse_expedition(content, boat_name, filename)
        elif ext == "csv":
            return cls._parse_csv(content, boat_name, filename)
        elif ext == "json":
            return cls._parse_orc_json(content, boat_name, filename)
        raise ValueError(f"Unsupported polar format: .{ext}  (accepted: .pol .txt .csv .json)")

    # ── Expedition ────────────────────────────────────────────────
    @classmethod
    def _parse_expedition(cls, content: str, boat_name: str, filename: str) -> PolarData:
        polar  = PolarData(boat_name=boat_name, source_file=filename)
        lines  = [l.strip() for l in content.splitlines() if l.strip()]

        # Find the header row (contains "twa" or "tws")
        header_idx = next(
            (i for i, l in enumerate(lines) if "twa" in l.lower() or "tws" in l.lower()),
            0,
        )
        header = lines[header_idx].replace(",", "\t").split()
        polar.tws_values = [float(v) for v in header[1:]]

        for line in lines[header_idx + 1:]:
            parts = line.replace(",", "\t").split()
            if not parts:
                continue
            try:
                twa = float(parts[0])
                row = {}
                for i, tws in enumerate(polar.tws_values):
                    if i + 1 < len(parts):
                        row[tws] = float(parts[i + 1])
                polar.twa_values.append(twa)
                polar.bsp_matrix[twa] = row
            except (ValueError, IndexError):
                continue

        polar.compute_vmg_angles()
        return polar

    # ── CSV ───────────────────────────────────────────────────────
    @classmethod
    def _parse_csv(cls, content: str, boat_name: str, filename: str) -> PolarData:
        polar  = PolarData(boat_name=boat_name, source_file=filename)
        reader = csv.reader(io.StringIO(content))
        rows   = [r for r in reader if any(c.strip() for c in r)]

        if not rows:
            raise ValueError("CSV file is empty")

        header = rows[0]
        polar.tws_values = [float(v) for v in header[1:] if v.strip()]

        for row in rows[1:]:
            if not row:
                continue
            try:
                twa = float(row[0])
                polar.twa_values.append(twa)
                polar.bsp_matrix[twa] = {
                    tws: float(row[i + 1])
                    for i, tws in enumerate(polar.tws_values)
                    if i + 1 < len(row) and row[i + 1].strip()
                }
            except (ValueError, IndexError):
                continue

        polar.compute_vmg_angles()
        return polar

    # ── ORC JSON ──────────────────────────────────────────────────
    @classmethod
    def _parse_orc_json(cls, content: str, boat_name: str, filename: str) -> PolarData:
        data = json.loads(content)
        polar = PolarData(boat_name=boat_name, source_file=filename)

        polar.tws_values = data["speeds"]
        polar.twa_values = data["angles"]

        for i, twa in enumerate(polar.twa_values):
            polar.bsp_matrix[twa] = {
                tws: data["values"][i][j]
                for j, tws in enumerate(polar.tws_values)
            }

        polar.compute_vmg_angles()
        return polar
