"""
captioner.py
------------
Generates rich English captions for all scraped Malaysian images using
a Gemma 4 multimodal model served by a llama.cpp HTTP server.

The llama.cpp server exposes an OpenAI-compatible REST API, so we use
the `openai` Python client pointed at the local server URL.

Prerequisites:
  1. Start your llama.cpp server:
       llama-server.exe -m <gemma4-mmproj-model.gguf> --port 8080
  2. Install requirements:
       pip install -r requirements.txt

Usage:
    python captioner.py [--images-dir images] [--output dataset.jsonl] [--output-txt dataset.txt]
                        [--base-url http://localhost:8080]
                        [--model gemma-4]
                        [--overwrite]
"""

import argparse
import base64
import json
import logging
import re
import sys
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Supported image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Captioning prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a professional image annotator for a Malaysian cultural dataset. "
    "Generate EXACTLY 5 distinct English captions for each image.\n\n"
    "Each caption must:\n"
    "- Describe Malaysian culture, place, food, people, or nature if visible.\n"
    "- Be factual, visual, and specific (colors, objects, setting, activity).\n"
    "- Be different in focus and wording from the others.\n"
    "- Be 1 sentences long.\n"
    "- NEVER start with 'This image shows' or similar phrases.\n\n"
    "Output rules:\n"
    "- Return ONLY the 5 captions.\n"
    "- One caption per line.\n"
    "- No numbering, bullets, or extra text."
)

USER_PROMPT = (
    "Please provide 5 distinct, detailed, and accurate captions for this image. "
    "Focus on anything that is specifically Malaysian. Separate each caption with a new line."
)

# Sentence-ending punctuation — used to detect truncated captions
_SENTENCE_ENDINGS = re.compile(r'[.!?]\s*$')


def strip_markdown(text: str) -> str:
    """Remove common markdown formatting tokens from a string."""
    # Bold / italic: **text**, *text*, __text__, _text_
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'\1', text)
    # Headings
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Bullet points / numbered lists
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # Collapse excess whitespace
    text = re.sub(r'\n{2,}', ' ', text)
    return text.strip()


def encode_image_base64(image_path: Path, max_size: int = 1024) -> tuple[str, str]:
    """
    Open, optionally downscale, and base64-encode an image.
    Returns (base64_string, mime_type).
    """
    with Image.open(image_path) as img:
        # Convert RGBA / palette images to RGB
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Downscale if too large to save on tokens / bandwidth
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        from io import BytesIO

        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")

    return encoded, "image/jpeg"


def caption_image(client: OpenAI, model: str, image_path: Path) -> list[str] | None:
    """Send one image to the llama.cpp Gemma 4 server and return a list of 5 captions.

    Returns None if the model returns an empty or truncated response.
    """
    b64, mime = encode_image_base64(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        max_tokens=1500,   # Raised to accommodate 5 captions
        temperature=0.5,   # Raised slightly to encourage variety between the 5 captions
    )

    choice = response.choices[0]
    raw = choice.message.content
    finish_reason = choice.finish_reason

    logger.debug(
        "Raw response for %s (finish_reason=%s): %r",
        image_path.name, finish_reason, raw,
    )

    if finish_reason == "length":
        logger.warning(
            "Caption truncated (hit max_tokens) for %s — will retry.",
            image_path.name,
        )
        return None

    if not raw:
        return None

    # Process the raw output into individual lines, stripping markdown/numbers
    raw_lines = raw.strip().split('\n')
    captions = []
    
    for line in raw_lines:
        cleaned = strip_markdown(line)
        if cleaned:
            captions.append(cleaned)

    if not captions:
        logger.warning("Empty captions received for %s", image_path.name)
        return None

    # Warn if the last caption doesn't end with sentence-closing punctuation
    if not _SENTENCE_ENDINGS.search(captions[-1]):
        logger.warning(
            "Last caption for %s may be incomplete (no sentence-ending punctuation): %r",
            image_path.name,
            captions[-1],
        )
        return None

    # We asked for 5, but the model might give slightly more or less.
    # To be safe, let's limit to the first 5 if it over-generated, or just return what it gave.
    return captions[:5]


def collect_images(images_dir: Path) -> list[Path]:
    """Recursively collect all valid image files under images_dir."""
    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_existing_captions(output_path: Path) -> set[str]:
    """Load already-captioned image paths from an existing JSONL file."""
    done: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Check for either the new list format or old single format
                    if "captions" in record and record["captions"]:
                        done.add(record["image"])
                    elif record.get("caption", "").strip():
                        done.add(record["image"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Caption Malaysian images using Gemma 4 via llama.cpp OpenAI-compatible server."
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="images",
        help="Directory containing scraped images (default: images/).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="dataset.jsonl",
        help="Output JSONL file path (default: dataset.jsonl).",
    )
    parser.add_argument(
        "--output-txt",
        type=str,
        default="dataset.txt",
        help="Output TXT file path for flat list (default: dataset.txt).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://127.0.0.1:8080/v1",
        help="Base URL for the llama.cpp OpenAI-compatible server (default: http://127.0.0.1:8080/v1).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-4",
        help="Model name to pass to the API (default: gemma-4).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-caption images even if they already exist in the output file.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=2.0,
        help="Seconds to wait before retrying a failed request (default: 2.0).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries per image (default: 3).",
    )
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    output_path = Path(args.output)
    output_txt_path = Path(args.output_txt)

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    all_images = collect_images(images_dir)
    if not all_images:
        logger.error("No images found in %s", images_dir)
        sys.exit(1)

    logger.info("Found %d images in %s", len(all_images), images_dir)

    already_done: set[str] = set()
    if not args.overwrite:
        already_done = load_existing_captions(output_path)
        logger.info("Skipping %d already-captioned images.", len(already_done))

    client = OpenAI(
        base_url=args.base_url,
        api_key="sk-no-key-required",
    )

    logger.info("Connecting to llama.cpp server at %s", args.base_url)
    logger.info("Using model: %s", args.model)

    successes = 0
    errors = 0

    # Open both JSONL and TXT files in append mode
    with output_path.open("a", encoding="utf-8") as out_json, \
         output_txt_path.open("a", encoding="utf-8") as out_txt:
        
        for img_path in tqdm(all_images, desc="Captioning", unit="img"):
            relative_path = img_path.as_posix()

            if relative_path in already_done:
                continue

            try:
                with Image.open(img_path) as _:
                    pass
            except UnidentifiedImageError:
                logger.warning("Skipping unreadable image: %s", img_path)
                errors += 1
                continue

            captions = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    captions = caption_image(client, args.model, img_path)
                    if captions:
                        break
                except Exception as exc:
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt, args.max_retries, img_path.name, exc,
                    )
                    if attempt < args.max_retries:
                        time.sleep(args.retry_delay * attempt)

            if not captions:
                logger.error("Failed to caption: %s", img_path)
                errors += 1
                continue

            parts = img_path.parts
            try:
                images_idx = parts.index(images_dir.name)
                category = parts[images_idx + 1] if images_idx + 1 < len(parts) else "unknown"
                keyword_folder = parts[images_idx + 2] if images_idx + 2 < len(parts) else "unknown"
            except ValueError:
                category = "unknown"
                keyword_folder = "unknown"

            # 1. Write to JSONL File
            record = {
                "image": relative_path,
                "captions": captions, # Changed from single string to a list of strings
                "category": category,
                "keyword": keyword_folder.replace("_", " "),
            }
            out_json.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_json.flush()

            # 2. Write to the extra TXT File
            for cap in captions:
                # Format: filename Caption 1...
                out_txt.write(f"{img_path.name} {cap}\n")
            out_txt.flush()
            
            successes += 1

    logger.info(
        "Captioning complete. Successes: %d | Errors: %d",
        successes, errors
    )
    logger.info("JSON Output saved to: %s", output_path.resolve())
    logger.info("TXT Output saved to: %s", output_txt_path.resolve())

if __name__ == "__main__":
    main()