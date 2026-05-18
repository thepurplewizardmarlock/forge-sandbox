# tanker-ais-experiment-1

Experimental tanker AIS pipeline. Streams live AIS messages, filters for
tanker vessels (ITU ship type 80-89), and prints position updates.

Will be moved to its own repository once validated.

## Data source

Currently uses **aisstream.io** as a free placeholder feed. Once the pipeline
shape is validated, the source will be swapped for **Kpler** by replacing
`sources/aisstream.py` with a Kpler equivalent that yields the same dict
shape.

## Setup

```
pip install -r requirements.txt
export AISSTREAM_API_KEY=your_key_here   # get one at https://aisstream.io
```

## Run

```
python main.py
```

You'll see lines like `TANKER 477123456 lat=1.234 lon=103.456 sog=12.3` as
tankers report positions. The first few minutes may be quiet — tanker
identification requires receiving a `ShipStaticData` message for each MMSI
before its position reports are recognized.

## Layout

```
main.py                  pipeline entry point
sources/aisstream.py     WebSocket source (placeholder)
sources/                 future: kpler.py
```
