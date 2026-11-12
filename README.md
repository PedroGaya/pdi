# PDI

## uv

**Windows:**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS / Linux:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Run

```bash
uv sync
uv run main.py
```

## Usage

Edit `PIPELINES` in `main.py`.

```python
PIPELINES = {
    "identity": identity,
    "clahe":    clahe,
}
```
