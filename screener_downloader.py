#!/usr/bin/env python3

import argparse
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.screener.in"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# 🔥 FIXED KEYWORDS
ANNUAL_KW = [
    "annual report",
    "financial year",
    "fy",
    "integrated report",
    "annual accounts"
]

CONCALL_KW = [
    "transcript",
    "earnings call",
    "concall"
]


# ─────────────────────────────────────────────
# Fetch page
# ─────────────────────────────────────────────
def fetch_page(session, symbol):
    for path in [f"/company/{symbol}/consolidated/", f"/company/{symbol}/"]:
        url = BASE_URL + path
        print(f"  → GET {url}")
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "lxml"), url
    raise SystemExit("❌ Failed to fetch company page")


# ─────────────────────────────────────────────
# Extract year
# ─────────────────────────────────────────────
def extract_year(text):
    t = text.lower()

    m = re.search(r"q[1-4][\s\-]*fy[\s\-]*(\d{2,4})", t)
    if m:
        yr = int(m.group(1))
        return 2000 + yr if yr < 100 else yr

    m = re.search(r"\bfy[\s\-]*(\d{2,4})", t)
    if m:
        yr = int(m.group(1))
        return 2000 + yr if yr < 100 else yr

    m = re.search(r"(20\d{2})[–\-/](\d{2,4})", t)
    if m:
        return int(m.group(1)) + 1

    m = re.search(r"(20\d{2})", t)
    if m:
        return int(m.group(1))

    return None


# ─────────────────────────────────────────────
# Classification (FIXED)
# ─────────────────────────────────────────────
def classify_doc(title, url):
    text = (title + " " + url).lower()

    # Annual reports (broad detection)
    if any(k in text for k in ANNUAL_KW):
        if "transcript" not in text:
            return "annual_report"

    # Concall
    if any(k in text for k in CONCALL_KW):
        if "board meeting" in text:
            return "other"
        return "concall"

    return "other"


# ─────────────────────────────────────────────
# Extract documents
# ─────────────────────────────────────────────
def extract_documents(soup, page_url):
    section = soup.find(id="documents")
    if not section:
        return []

    docs = []
    seen = set()

    for a in section.find_all("a", href=True):
        href = a["href"].strip()
        title = a.get_text(" ", strip=True)  # FIXED spacing

        if href.startswith("/"):
            href = urljoin(BASE_URL, href)

        if not href.startswith("http"):
            continue

        full_text = f"{title} {href}"
        year = extract_year(full_text)
        doc_type = classify_doc(title, href)

        if doc_type == "other":
            continue

        key = (doc_type, year, title.lower())
        if key in seen:
            continue
        seen.add(key)

        docs.append({
            "title": title or "document",
            "url": href,
            "year": year,
            "doc_type": doc_type,
        })

    return docs


# ─────────────────────────────────────────────
# Download (ROBUST)
# ─────────────────────────────────────────────
def safe_name(s):
    return re.sub(r"[^\w\-_. ]", "_", s)[:100]


def download_pdf(session, doc, out_root):
    url = doc["url"]
    year = str(doc["year"] or "unknown")
    dtype = doc["doc_type"]
    title = safe_name(doc["title"])

    dest = out_root / dtype / year
    dest.mkdir(parents=True, exist_ok=True)

    fpath = dest / f"{year}_{title}.pdf"

    print(f"[↓] {dtype} {year} {title}")

    try:
        headers = session.headers.copy()

        # 🔥 Critical for BSE/NSE
        if "bseindia.com" in url:
            headers["Referer"] = "https://www.bseindia.com/"
        elif "nseindia.com" in url:
            headers["Referer"] = "https://www.nseindia.com/"

        r = session.get(url, timeout=60, stream=True, headers=headers)
        r.raise_for_status()

        # ❌ detect HTML instead of PDF
        content_type = r.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            print("    ⚠ skipped (HTML page)")
            return False

        with open(fpath, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)

        size_kb = fpath.stat().st_size // 1024

        if size_kb < 10:
            fpath.unlink(missing_ok=True)
            print("    ⚠ skipped (too small)")
            return False

        print(f"    ✓ saved ({size_kb} KB)")
        return True

    except Exception as e:
        print(f"    ✗ failed: {e}")
        return False


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    symbol = args.symbol.upper()

    session = requests.Session()
    session.headers.update(HEADERS)

    print("\n[1] Fetching page")
    soup, page_url = fetch_page(session, symbol)

    print("[2] Extracting documents")
    docs = extract_documents(soup, page_url)

    print(f"✔ Clean docs found: {len(docs)}\n")

    for d in docs:
        print(f"{d['doc_type']} | {d['year']} | {d['title']}")

    if args.dry_run:
        return

    out_dir = Path("./screener_docs") / symbol

    print("\n[3] Downloading...\n")

    success = 0
    fail = 0

    for doc in docs:
        if download_pdf(session, doc, out_dir):
            success += 1
        else:
            fail += 1
        time.sleep(1.5)

    print("\n==============================")
    print(f"Downloaded: {success}")
    print(f"Failed: {fail}")
    print("==============================\n")


if __name__ == "__main__":
    main()
#python screener_downloader.py ADANIPORTS run cmd