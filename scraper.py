"""
scraper.py
----------
Scrapes Malaysia-related images from Google Images (via Bing as a fallback)
using the icrawler library.

Output layout:
    images/
        <category>/
            000001.jpg
            000002.jpg
            ...

Usage:
    python scraper.py [--max-num N]
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from icrawler.builtin import BingImageCrawler, GoogleImageCrawler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keyword catalogue – categories of Malaysian subjects
# ---------------------------------------------------------------------------
KEYWORDS: dict[str, list[str]] = {
    "landmarks": [
        "Petronas Twin Towers Kuala Lumpur",
        "Batu Caves Malaysia",
        "Penang George Town street art",
        "A Famosa fort Melaka Malaysia",
        "Langkawi Sky Bridge Malaysia",
        "Menara KL Tower night",
        "Putrajaya mosque Malaysia",
        "Kota Kinabalu waterfront Sabah",
        "Kuching waterfront Sarawak",
        "Cameron Highlands Malaysia",
    ],
    "food": [
        "Nasi Lemak Malaysia",
        "Char Kway Teow Penang",
        "Roti Canai Malaysia",
        "Laksa Sarawak",
        "Satay Malaysia grilled",
        "Cendol Malaysia dessert",
        "Rendang Malaysia beef",
        "Wantan Mee Malaysia noodles",
        "Apam Balik Malaysia pancake",
        "Durian Malaysia fruit",
    ],
    "nature": [
        "Taman Negara rainforest Malaysia",
        "Mount Kinabalu Sabah Malaysia",
        "Danum Valley Borneo Malaysia",
        "Proboscis monkey Borneo",
        "Orangutan Sepilok Sabah Malaysia",
        "Malayan tiger Malaysia wildlife",
        "Perhentian Islands Malaysia beach",
        "Mulu Caves Sarawak Malaysia",
        "Sekinchan paddy fields Malaysia",
        "Mossy Forest Cameron Highlands",
    ],
    "culture": [
        "Malaysia traditional Batik fabric",
        "Malay traditional house kampung",
        "Hari Raya celebration Malaysia",
        "Chinese New Year Malaysia parade",
        "Deepavali light festival Malaysia",
        "Gawai harvest festival Sarawak",
        "Wau Bulan Malaysian kite",
        "Silat martial art Malaysia",
        "Malaysia national costume Baju Melayu",
        "Orang Asli indigenous Malaysia",
    ],
    "cityscape": [
        "Kuala Lumpur skyline night",
        "Penang Hill view Malaysia",
        "Johor Bahru city Malaysia",
        "Ipoh old town Malaysia",
        "Kota Bharu Kelantan Malaysia market",
    ],
}


def crawl_category(
    category: str,
    keywords: list[str],
    images_root: Path,
    max_num: int,
    use_bing: bool = False,
) -> None:
    """Download images for every keyword inside a given category."""
    cat_dir = images_root / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    CrawlerClass = BingImageCrawler if use_bing else GoogleImageCrawler

    for keyword in keywords:
        # Build a safe sub-folder name from the keyword
        safe_kw = keyword.lower().replace(" ", "_")[:60]
        kw_dir = cat_dir / safe_kw
        kw_dir.mkdir(parents=True, exist_ok=True)

        logger.info("[%s] Crawling: '%s'  →  %s", category, keyword, kw_dir)

        crawler = CrawlerClass(
            feeder_threads=1,
            parser_threads=2,
            downloader_threads=4,
            storage={"root_dir": str(kw_dir)},
        )

        filters = dict(type="photo", size="large")

        try:
            crawler.crawl(
                keyword=keyword,
                filters=filters,
                max_num=max_num,
                min_size=(300, 300),
                file_idx_offset=0,
            )
        except Exception as exc:
            logger.warning("Error crawling '%s': %s", keyword, exc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Malaysian images from Google/Bing Images."
    )
    parser.add_argument(
        "--max-num",
        type=int,
        default=30,
        help="Maximum number of images to download per keyword (default: 30).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images",
        help="Root directory to save images (default: images/).",
    )
    parser.add_argument(
        "--use-bing",
        action="store_true",
        help="Use Bing Image crawler instead of Google (more reliable).",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=list(KEYWORDS.keys()),
        help="Only scrape a specific category (default: all).",
    )
    args = parser.parse_args()

    images_root = Path(args.output_dir)
    images_root.mkdir(parents=True, exist_ok=True)

    categories = (
        {args.category: KEYWORDS[args.category]}
        if args.category
        else KEYWORDS
    )

    total_kw = sum(len(v) for v in categories.values())
    logger.info(
        "Starting scraper  |  categories=%d  keywords=%d  max_num=%d  engine=%s",
        len(categories),
        total_kw,
        args.max_num,
        "Bing" if args.use_bing else "Google",
    )

    for category, keywords in categories.items():
        crawl_category(
            category=category,
            keywords=keywords,
            images_root=images_root,
            max_num=args.max_num,
            use_bing=args.use_bing,
        )

    logger.info("Scraping complete. Images stored in: %s", images_root.resolve())


if __name__ == "__main__":
    main()
