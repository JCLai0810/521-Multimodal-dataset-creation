# Malaysia Image Dataset

A curated image dataset of Malaysian landmarks, food, nature, culture, and cityscapes with AI-generated captions produced by **Gemma 4** (via llama.cpp).

---

## Project Structure

```
521Assignment1/
├── images/                  # Downloaded images (created by scraper.py)
│   ├── landmarks/
│   │   └── <keyword>/
│   ├── food/
│   ├── nature/
│   ├── culture/
│   └── cityscape/
├── dataset.jsonl            # Final dataset (image paths + captions)
├── scraper.py               # Step 1: Download images
├── captioner.py             # Step 2: Generate captions with Gemma 4
├── requirements.txt         # Python dependencies
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Step 1 — Scrape Images

```bash
# Scrape all categories, 30 images per keyword (default)
python scraper.py

# Scrape using Bing (more reliable, fewer blocks)
python scraper.py --use-bing

# Only scrape the food category, 50 images per keyword
python scraper.py --category food --max-num 50

# Full list of options
python scraper.py --help
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--max-num` | 30 | Images to download per keyword |
| `--output-dir` | `images/` | Root image directory |
| `--use-bing` | False | Use Bing crawler instead of Google |
| `--category` | all | Scrape one specific category |

---

## Step 2 — Start llama.cpp Server

Download a Gemma 4 multimodal GGUF model and launch the server:

```bash
# Windows (PowerShell)
.\llama-server.exe `
  --model gemma-4b-it-Q4_K_M.gguf `
  --mmproj gemma-4b-mmproj.gguf `
  --port 8080 `
  --ctx-size 8192
```

> The server will be accessible at `http://localhost:8080/v1` with an OpenAI-compatible API.

---

## Step 3 — Caption Images

```bash
# Caption all images (default: connects to localhost:8080)
python captioner.py

# Custom server URL or model name
python captioner.py --base-url http://localhost:8080/v1 --model gemma-4

# Re-caption everything from scratch
python captioner.py --overwrite

# Full list of options
python captioner.py --help
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--images-dir` | `images/` | Directory with scraped images |
| `--output` | `dataset.jsonl` | Output JSONL file |
| `--base-url` | `http://localhost:8080/v1` | llama.cpp server URL |
| `--model` | `gemma-4` | Model name sent in API calls |
| `--overwrite` | False | Re-caption already-processed images |
| `--max-retries` | 3 | Retries per failed image |
| `--retry-delay` | 2.0 | Seconds between retries |

The captioner **automatically resumes** if interrupted — it skips images already present in `dataset.jsonl`.

---

## Dataset Format (`dataset.jsonl`)

Each line is a JSON object:

```json
{
  "image": "images/landmarks/petronas_twin_towers_kuala_lumpur/000001.jpg",
  "caption": "The iconic Petronas Twin Towers rise dramatically into the evening sky above Kuala Lumpur, their steel-and-glass facades reflecting warm golden light...",
  "category": "landmarks",
  "keyword": "Petronas Twin Towers Kuala Lumpur"
}
```

---

## Dataset Categories & Keywords

| Category | Count |
|---|---|
| Landmarks | 10 keywords |
| Food | 10 keywords |
| Nature | 10 keywords |
| Culture | 10 keywords |
| Cityscape | 5 keywords |

At 30 images/keyword → **~1,350 images** total (before deduplication/filtering).

---

## Notes

- **Google scraping** may occasionally be rate-limited. Use `--use-bing` if you encounter issues.
- The captioner sends images at a **maximum resolution of 1024 px** to balance quality and speed.
- Captions are written incrementally to `dataset.jsonl` so the pipeline is **safe to interrupt and resume**.
