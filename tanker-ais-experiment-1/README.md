# tanker-ais-experiment-1

Experimental tanker AIS pipeline. Runs as a **snapshot job** — connects to
aisstream.io for a fixed window, captures every position report from tanker
vessels (ITU ship type 80-89) during that window, and writes one JSONL
snapshot file per run. Designed to be cron-driven a handful of times per
day rather than run continuously.

Will be moved to its own repository once validated.

## Data source

Currently **aisstream.io** as a free placeholder feed. Once validated, the
source will be swapped for **Kpler** by replacing `sources/aisstream.py`
with a Kpler equivalent that yields the same message-dict shape. The
snapshot logic and output format stay unchanged.

## Setup

```
pip install -r requirements.txt
export AISSTREAM_API_KEY=your_key_here   # get one at https://aisstream.io
```

## Run a single snapshot

```
python main.py                       # 15-minute window (default)
python main.py --window-minutes 20   # longer window for sparser regions
```

Each run writes `snapshots/tanker-snapshot-<UTC>.jsonl` (one position
report per line, raw aisstream message) and updates
`cache/mmsi_to_type.json` (persistent MMSI → ship-type map).

## How much data per ship per day

Conservative coverage rationale (based on ITU-R M.1371 broadcast intervals):

- **Position reports** — Class A vessels broadcast every ~3 min when
  anchored/moored, every 2–10 sec when underway. A 15-min window captures
  at least ~5 reports per anchored tanker and 30+ per moving tanker.
- **Ship static data** (carries the type code that identifies a tanker) —
  broadcast every ~6 min. A 15-min window is 2.5× that interval, so most
  active tankers are identified within a single run; the persistent cache
  ensures any tanker seen on any prior run is recognized immediately on
  the next.
- **Recommended cadence**: 4–6 runs/day (every 4–6 hours via cron) gives
  each cached tanker ~5–30+ position fixes/day even when anchored, and
  smooths over satellite/coastal coverage gaps.

The first 24-48 hours of runs are mainly bootstrapping the MMSI cache;
coverage stabilizes after that.

### Example cron (every 4 hours)

```
0 */4 * * * cd /path/to/tanker-ais-experiment-1 && \
  AISSTREAM_API_KEY=... /usr/bin/python3 main.py >> run.log 2>&1
```

## Layout

```
main.py                  snapshot job entry point
sources/aisstream.py     aisstream.io WebSocket source (placeholder)
sources/                 future home for kpler.py
cache/                   persistent MMSI -> ship-type map (gitignored)
snapshots/               JSONL snapshot files, one per run (gitignored)
```
