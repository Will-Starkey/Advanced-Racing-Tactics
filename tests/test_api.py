"""
Integration tests for the FastAPI endpoints.
No real SignalK connection — instruments state is mocked.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


# ── App setup ─────────────────────────────────────────────────────
# Patch lifespan so SignalK/asyncio don't run during tests
import main as app_module

@pytest.fixture(scope="module")
def client():
    with patch.object(app_module, "brand", app_module.InstrumentBrand.UNKNOWN), \
         patch.object(app_module, "adapter", None):
        from fastapi.testclient import TestClient
        # Override lifespan to be a no-op
        from contextlib import asynccontextmanager
        @asynccontextmanager
        async def noop_lifespan(app):
            yield
        app_module.app.router.lifespan_context = noop_lifespan
        with TestClient(app_module.app) as c:
            yield c


# ── Static files ──────────────────────────────────────────────────

class TestStaticFiles:
    def test_index_html_served(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]

    def test_upload_html_served(self, client):
        r = client.get("/upload.html")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


# ── /tactics ──────────────────────────────────────────────────────

class TestTacticsEndpoint:
    def test_returns_waiting_when_no_instruments(self, client):
        r = client.get("/tactics")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "waiting for instruments"

    def test_returns_live_data_when_instruments_present(self, client):
        from instruments.base import InstrumentData
        from tactics_engine import TacticalState

        mock_inst = InstrumentData(twd=260, tws=12, heading=218, bsp=7,
                                   sog=7, awa=42, heel=5, lat=41.5, lon=-71.3)
        mock_state = TacticalState(
            shift_type="lift", shift_degrees=5.0,
            shift_state="transient", shift_age_seconds=20.0,
            wind_trend="steady", twa=42.0, on_starboard=True,
            vmg=5.2, optimal_twa=42.0, ttm_minutes=3.5,
            tack_recommendation="hold",
            tack_recommendation_reason="transient shift",
            port_layline=302.0, stbd_layline=218.0,
        )

        with patch.object(app_module, "latest_inst", mock_inst), \
             patch.object(app_module, "latest_state", mock_state), \
             patch.object(app_module, "latest_advice", "Hold starboard."):
            r = client.get("/tactics")

        assert r.status_code == 200
        data = r.json()
        assert "status" not in data
        assert data["tws"] == 12.0
        assert data["shift"] == "lift"
        assert data["tack_recommendation"] == "hold"
        assert data["advice"] == "Hold starboard."


# ── /polars/upload ────────────────────────────────────────────────

VALID_CSV = """\
TWA,6,8,10,12,16
42,5.40,6.51,7.04,7.30,7.52
52,5.70,6.80,7.28,7.51,7.73
90,7.07,7.92,8.52,9.02,9.71
150,5.13,6.58,7.62,8.35,9.73
"""

class TestPolarUpload:
    def test_valid_csv_upload(self, client):
        r = client.post(
            "/polars/upload",
            data={"boat_name": "TestBoat"},
            files={"file": ("polar.csv", VALID_CSV, "text/csv")},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "uploaded"
        assert data["boat"] == "TestBoat"
        assert "tws_range" in data
        assert "upwind_angles" in data

    def test_unsupported_format_returns_400(self, client):
        r = client.post(
            "/polars/upload",
            data={"boat_name": "TestBoat"},
            files={"file": ("polar.pdf", b"garbage", "application/pdf")},
        )
        assert r.status_code == 400

    def test_missing_boat_name_returns_422(self, client):
        r = client.post(
            "/polars/upload",
            files={"file": ("polar.csv", VALID_CSV, "text/csv")},
        )
        assert r.status_code == 422


# ── /polars/activate ──────────────────────────────────────────────

class TestPolarActivate:
    def test_activate_uploaded_boat(self, client):
        # Upload first
        client.post(
            "/polars/upload",
            data={"boat_name": "ActivateMe"},
            files={"file": ("polar.csv", VALID_CSV, "text/csv")},
        )
        r = client.post("/polars/activate", data={"boat_name": "ActivateMe"})
        assert r.status_code == 200
        assert r.json()["status"] == "active"

    def test_activate_nonexistent_returns_404(self, client):
        r = client.post("/polars/activate", data={"boat_name": "NoSuchBoat"})
        assert r.status_code == 404


# ── /polars/list ──────────────────────────────────────────────────

class TestPolarList:
    def test_list_includes_uploaded_boat(self, client):
        client.post(
            "/polars/upload",
            data={"boat_name": "ListMe"},
            files={"file": ("polar.csv", VALID_CSV, "text/csv")},
        )
        r = client.get("/polars")
        assert r.status_code == 200
        assert "listme" in r.json()["boats"]


# ── /mark ─────────────────────────────────────────────────────────

class TestMarkEndpoint:
    def test_set_mark_valid(self, client):
        r = client.post("/mark", data={"lat": "41.501", "lon": "-71.318"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "ok"
        assert abs(data["lat"] - 41.501) < 0.0001

    def test_set_mark_invalid_returns_422(self, client):
        r = client.post("/mark", data={"lat": "not_a_float", "lon": "-71.318"})
        assert r.status_code == 422


# ── /status ───────────────────────────────────────────────────────

class TestStatusEndpoint:
    def test_status_returns_json(self, client):
        r = client.get("/status")
        assert r.status_code == 200
        data = r.json()
        assert "instruments" in data
        assert "brand" in data
