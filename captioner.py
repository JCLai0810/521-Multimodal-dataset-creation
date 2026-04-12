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
    python captioner.py [--images-dir images] [--output dataset.jsonl]
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
    "You are an expert annotator for a Malaysian cultural and geographical image dataset. "
    "Your task is to produce a single, detailed, factual English caption for each image. "
    "The caption should:\n"
    "  • Identify the specific Malaysian location, food, cultural element, or wildlife shown.\n"
    "  • Describe visual attributes: colours, textures, composition, time of day, weather.\n"
    "  • Mention any recognisable landmarks, signs, people (without identifying them), or activities.\n"
    "  • Be between 2 and 5 sentences long.\n"
    "  • Never start with 'This image shows' or 'The image depicts'.\n"
    "Return ONLY the caption text, no preamble or JSON."
)

USER_PROMPT = (
    "Please provide a detailed and accurate caption for this image. "
    "Focus on anything that is specifically Malaysian."
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


def caption_image(client: OpenAI, model: str, image_path: Path) -> str | None:
    """Send one image to the llama.cpp Gemma 4 server and return the caption.

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
                        # NOTE: do NOT pass 'detail' – llama.cpp does not
                        # support it and returns an empty completion.
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                        },
                    },
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ],
        max_tokens=1024,   # raised from 512 to avoid mid-sentence cutoff
        temperature=0.3,
    )

    choice = response.choices[0]
    raw = choice.message.content
    finish_reason = choice.finish_reason

    logger.debug(
        "Raw response for %s (finish_reason=%s): %r",
        image_path.name, finish_reason, raw,
    )

    # finish_reason == 'length' means the token budget was exhausted
    # before the model finished the sentence — treat as failure.
    if finish_reason == "length":
        logger.warning(
            "Caption truncated (hit max_tokens) for %s — will retry.",
            image_path.name,
        )
        return None

    caption = strip_markdown((raw or "").strip())

    if not caption:
        logger.warning(
            "Empty caption received for %s — model returned: %r",
            image_path.name,
            raw,
        )
        return None

    # Warn if the caption doesn't end with sentence-closing punctuation
    # (may indicate a subtle truncation llama.cpp didn't report as 'length')
    if not _SENTENCE_ENDINGS.search(caption):
        logger.warning(
            "Caption for %s may be incomplete (no sentence-ending punctuation): %r",
            image_path.name,
            caption,
        )
        return None

    return caption


def collect_images(images_dir: Path) -> list[Path]:
    """Recursively collect all valid image files under images_dir."""
    return sorted(
        p
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_existing_captions(output_path: Path) -> set[str]:
    """Load already-captioned image paths from an existing JSONL file.

    Only records with a non-empty caption are considered done.
    Records with empty captions (from a previous failed run) will be
    re-processed.
    """
    done: set[str] = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("caption", "").strip():
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
        "--base-url",
        type=str,
        default="http://localhost:8080/v1",
        help="Base URL for the llama.cpp OpenAI-compatible server (default: http://localhost:8080/v1).",
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

    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    # Collect all images
    all_images = collect_images(images_dir)
    if not all_images:
        logger.error("No images found in %s", images_dir)
        sys.exit(1)

    logger.info("Found %d images in %s", len(all_images), images_dir)

    # Skip images already captioned (resume support)
    already_done: set[str] = set()
    if not args.overwrite:
        already_done = load_existing_captions(output_path)
        logger.info("Skipping %d already-captioned images.", len(already_done))

    # Build the OpenAI client pointing to the llama.cpp server
    client = OpenAI(
        base_url=args.base_url,
        api_key="sk-no-key-required",  # llama.cpp doesn't require a real key
    )

    logger.info("Connecting to llama.cpp server at %s", args.base_url)
    logger.info("Using model: %s", args.model)

    # Open output file in append mode so we can resume
    successes = 0
    errors = 0

    with output_path.open("a", encoding="utf-8") as out_f:
        for img_path in tqdm(all_images, desc="Captioning", unit="img"):
            # Use forward-slash relative path as the image key
            relative_path = img_path.as_posix()

            if relative_path in already_done:
                continue

            # Validate image before sending
            try:
                with Image.open(img_path) as _:
                    pass
            except UnidentifiedImageError:
                logger.warning("Skipping unreadable image: %s", img_path)
                errors += 1
                continue

            # Attempt captioning with retries
            caption = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    caption = caption_image(client, args.model, img_path)
                    break
                except Exception as exc:
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s",
                        attempt,
                        args.max_retries,
                        img_path.name,
                        exc,
                    )
                    if attempt < args.max_retries:
                        time.sleep(args.retry_delay * attempt)

            if caption is None:
                logger.error("Failed to caption: %s", img_path)
                errors += 1
                continue

            # Derive category from folder structure: images/<category>/<keyword>/file.jpg
            parts = img_path.parts
            try:
                images_idx = parts.index(images_dir.name)
                category = parts[images_idx + 1] if images_idx + 1 < len(parts) else "unknown"
                keyword_folder = parts[images_idx + 2] if images_idx + 2 < len(parts) else "unknown"
            except ValueError:
                category = "unknown"
                keyword_folder = "unknown"

            record = {
                "image": relative_path,
                "caption": caption,
                "category": category,
                "keyword": keyword_folder.replace("_", " "),
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            successes += 1

    logger.info(
        "Captioning complete. Successes: %d | Errors: %d | Output: %s",
        successes,
        errors,
        output_path.resolve(),
    )


if __name__ == "__main__":
    main()
