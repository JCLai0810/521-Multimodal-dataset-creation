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
import random
from pathlib import Path
from collections import defaultdict

from openai import OpenAI
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

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

_SENTENCE_ENDINGS = re.compile(r'[.!?]\s*$')


def strip_markdown(text: str) -> str:
    text = re.sub(r'\*{1,2}([^*]+)\*{1,2}', r'\1', text)
    text = re.sub(r'_{1,2}([^_]+)_{1,2}', r'\1', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    return text.strip()


def encode_image_base64(image_path: Path, max_size: int = 1024):
    with Image.open(image_path) as img:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        from io import BytesIO
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)

        return base64.b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def caption_image(client, model, image_path):
    b64, mime = encode_image_base64(image_path)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        max_tokens=1500,
        temperature=0.6,
    )

    raw = response.choices[0].message.content
    if not raw:
        return None

    lines = [strip_markdown(l) for l in raw.split("\n") if l.strip()]

    if len(lines) < 5:
        return None

    if not _SENTENCE_ENDINGS.search(lines[-1]):
        return None

    return lines[:5]


def collect_images(images_dir: Path):
    return sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="images")
    parser.add_argument("--output", default="dataset.jsonl")
    parser.add_argument("--output-txt", default="dataset.txt")
    parser.add_argument("--base-url", default="http://127.0.0.1:8081/v1")
    parser.add_argument("--model", default="gemma-4")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=2.0)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    all_images = collect_images(images_dir)

    if not all_images:
        logger.error("No images found in %s", images_dir)
        sys.exit(1)

    logger.info("Found %d images", len(all_images))

    # ✅ GROUP BY KEYWORD
    keyword_groups = defaultdict(list)

    for img_path in all_images:
        parts = img_path.parts
        try:
            idx = parts.index(images_dir.name)
            keyword = parts[idx + 2]
        except Exception:
            keyword = "unknown"

        keyword_groups[keyword].append(img_path)

    logger.info("Total keywords: %d", len(keyword_groups))

    client = OpenAI(
        base_url=args.base_url,
        api_key="sk-no-key-required"
    )

    successes = 0
    errors = 0

    with open(args.output, "w", encoding="utf-8") as out_json, \
         open(args.output_txt, "w", encoding="utf-8") as out_txt:

        for keyword, images in tqdm(keyword_groups.items(), desc="Keywords"):

            rep_image = random.choice(images)

            try:
                with Image.open(rep_image):
                    pass
            except UnidentifiedImageError:
                logger.warning("Bad image: %s", rep_image)
                continue

            captions = None

            for attempt in range(1, args.max_retries + 1):
                try:
                    captions = caption_image(client, args.model, rep_image)
                    if captions:
                        break
                except Exception as e:
                    logger.warning("Retry %d failed: %s", attempt, e)
                    time.sleep(args.retry_delay * attempt)

            if not captions:
                logger.error("Failed keyword: %s", keyword)
                errors += 1
                continue

            # Extract category
            parts = rep_image.parts
            try:
                idx = parts.index(images_dir.name)
                category = parts[idx + 1]
            except Exception:
                category = "unknown"

            # ✅ Assign captions to all images
            for img_path in images:
                record = {
                    "image": img_path.as_posix(),
                    "captions": captions,
                    "category": category,
                    "keyword": keyword.replace("_", " "),
                }

                out_json.write(json.dumps(record, ensure_ascii=False) + "\n")

                for cap in captions:
                    out_txt.write(f"{keyword}_{img_path.name} {cap}\n")

                successes += 1

    logger.info("Done. Success: %d | Errors: %d", successes, errors)


if __name__ == "__main__":
    main()