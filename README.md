# Advanced Sailing Tactics Display

Real-time sailing tactics system for racing. Connects to B&G or Garmin instruments
via NMEA 2000 / Signal K, computes tactical data, and uses Claude AI to generate
tactical advice. Displayed on a tablet in the cockpit.

---

## System Architecture

```
[B&G or Garmin Instruments]
        │ NMEA 2000 backbone
[Actisense NGT-1 USB gateway]
        │ USB
[Raspberry Pi 4]
  ├── Signal K Server  (port 3000)  ← normalises all instrument data
  ├── tactics_system/               ← this project
  │     ├── main.py  (FastAPI, port 8000)
  │     ├── tactics_engine.py
  │     ├── signalk_client.py
  │     ├── llm_bridge.py
  │     ├── polar/
  │     └── instruments/
  └── WiFi Hotspot
        │
[iPad / Tablet Browser]  →  http://10.42.0.1:8000
```

---

## Hardware Required

| Item | Purpose | ~Cost |
|------|---------|-------|
| Actisense NGT-1-USB | Tap into NMEA 2000 backbone | $200 |
| Raspberry Pi 4 (4 GB) | Run Signal K + tactics server | $90 |
| Waterproof case + USB battery | Power at sea | $50 |
| iPad (any model) | Cockpit display | — |

---

## Setup

### 1. Install Signal K on the Raspberry Pi

```bash
sudo apt update && sudo apt install -y nodejs npm
sudo npm install -g @signalk/server
signalk-server --sample-nmea0183data   # test with sample data
```

Connect the Actisense NGT-1 via USB and configure it as a data connection
in the Signal K admin UI at http://localhost:3000.

### 2. Clone and configure this project

```bash
git clone <repo> tactics_system
cd tactics_system
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY and set SIGNALK_HOST
```

### 3. Install Python dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the server

```bash
python main.py
```

The tablet UI is served at **http://\<pi-ip\>:8000**
The polar upload page is at **http://\<pi-ip\>:8000/upload.html**

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/tactics` | Live tactical state (instruments + computed) |
| GET | `/status` | System status (brand, instruments, polar, mark) |
| GET | `/polars` | List stored polars |
| POST | `/polars/upload` | Upload a new polar file |
| POST | `/polars/activate` | Set active polar by boat name |
| GET | `/polars/active` | Currently active polar info |
| POST | `/mark` | Set windward mark position (lat, lon) |
| GET | `/mark` | Get current mark position |

---

## Polar File Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Expedition | `.pol` or `.txt` | Most common racing software format |
| CSV | `.csv` | First row = TWS values, first col = TWA |
| ORC JSON | `.json` | Exported from ORC rating database |

---

## Instrument Support

| Feature | Garmin | B&G Standard | B&G H5000 |
|---------|--------|--------------|-----------|
| Wind / heading / speed | ✓ | ✓ | ✓ |
| GPS position | ✓ | ✓ | ✓ |
| Performance vs polar | computed | computed | native |
| Optimal VMG angle | from polar | from polar | native |
| Beat / gybe angles | from polar | from polar | native |
