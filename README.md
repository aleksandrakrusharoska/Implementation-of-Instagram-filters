# Instagram Filters — Python + OpenCV

## Filters

| Filter | Source |
|---|---|
| Clarendon, Gingham, Lark, Mayfair, Valencia, Walden | Exact CSSgram recipes |
| Sierra | Documented CSS recipe (designpieces.com) |
| Juno | Approximation (no public canonical recipe) |
| Rainbow | Custom procedural HSV layer |

## Requirements

```
pip install opencv-python numpy
```

## Usage

**Apply a single filter to an image:**
```
python instagram_filters.py image.jpg --filter clarendon -o output.jpg
```

**Apply all filters at once:**
```
python instagram_filters.py image.jpg --filter all -o results/
```

**Save a comparison grid of all filters:**
```
python instagram_filters.py image.jpg --grid -o comparison.jpg
```

**Live camera preview with real-time filter switching:**
```
python instagram_filters_live.py
```

## Live Camera Controls

| Input | Action |
|---|---|
| Left click — right half | Next filter |
| Left click — left half | Previous filter |
| Right click | Previous filter |
| `n` / `d` | Next filter |
| `p` / `a` | Previous filter |
| `0`–`9` | Jump to filter by index |
| `+` / `-` | Increase / decrease filter strength |
| `s` | Save snapshot |
| `g` | Save comparison grid |
| `q` / `ESC` | Quit |
