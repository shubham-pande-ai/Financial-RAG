#!/usr/bin/env python3

# screener_downloader -> upload_to_minio.py & db/database.py -> 
# ingest -> ( pdf_extractor -> chunker.py -> embedder -> qdrant_loader )  -> 
# server.py & app.py -> atomic_decomposer -> schema_bridge.py & retriever.py
# -> reranker.py -> fusion_layer.py -> prompt_builder.py -> rag_engine.py ->
# eval_suite.py
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
    # Try both consolidated and standalone company page URLs
    for path in [f"/company/{symbol}/consolidated/", f"/company/{symbol}/"]:
        url = BASE_URL + path
        print(f"  → GET {url}")
        r = session.get(url, timeout=30)
        # Return parsed HTML if the request is successful
        if r.status_code == 200:
            return BeautifulSoup(r.text, "lxml"), url
    # Exit if we cannot fetch the company page
    raise SystemExit("❌ Failed to fetch company page")


# ─────────────────────────────────────────────
# Extract year
# ─────────────────────────────────────────────
def extract_year(text):
    t = text.lower()

    # Match quarters with fiscal year, e.g., "Q1 FY23", "q4-fy2024", "q2 fy 22"
    m = re.search(r"q[1-4][\s\-]*fy[\s\-]*(\d{2,4})", t)
    if m:
        yr = int(m.group(1))
        return 2000 + yr if yr < 100 else yr

    # Match standalone fiscal year with word boundary, e.g., "FY23", "fy-2024", "FY 22"
    m = re.search(r"\bfy[\s\-]*(\d{2,4})", t)
    if m:
        yr = int(m.group(1))
        return 2000 + yr if yr < 100 else yr

    # Match financial year ranges, e.g., "2023-24", "2023-2024", "2023/24", extracting the start year and returning the end year
    m = re.search(r"(20\d{2})[–\-/](\d{2,4})", t)
    if m:
        return int(m.group(1)) + 1

    # Match any 4-digit year starting with 20, e.g., "2023"
    m = re.search(r"(20\d{2})", t)
    if m:
        return int(m.group(1))

    return None


# ─────────────────────────────────────────────
# Classification (FIXED)
# ─────────────────────────────────────────────
def classify_doc(title, url):
    # Combine title and URL to search for keywords in lowercase
    text = (title + " " + url).lower()

    # Identify annual reports by checking keywords and excluding transcripts
    if any(k in text for k in ANNUAL_KW):
        if "transcript" not in text:
            return "annual_report"

    # Identify concalls, excluding board meeting documents
    if any(k in text for k in CONCALL_KW):
        if "board meeting" in text:
            return "other"
        return "concall"

    # Default category for everything else
    return "other"


# ─────────────────────────────────────────────
# Extract documents
# ─────────────────────────────────────────────
def extract_documents(soup, page_url):
    # Locate the documents section on the page
    section = soup.find(id="documents")
    if not section:
        return []

    docs = []
    seen = set()

    # Iterate through all hyperlinks within the documents section
    for a in section.find_all("a", href=True):
        href = a["href"].strip()
        title = a.get_text(" ", strip=True)  # FIXED spacing

        # Make relative URLs absolute
        if href.startswith("/"):
            href = urljoin(BASE_URL, href)

        # Ignore non-http links
        if not href.startswith("http"):
            continue

        # Extract the year and document type from the link text and URL
        full_text = f"{title} {href}"
        year = extract_year(full_text)
        doc_type = classify_doc(title, href)

        # Skip documents we don't care about
        if doc_type == "other":
            continue

        # Prevent duplicate entries by tracking seen combinations
        key = (doc_type, year, title.lower())
        if key in seen:
            continue
        seen.add(key)

        # Append the structured document metadata
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
    # Extract document metadata
    url = doc["url"]
    year = str(doc["year"] or "unknown")
    dtype = doc["doc_type"]
    title = safe_name(doc["title"])

    # Create destination directory grouped by doc type and year
    dest = out_root / dtype / year
    dest.mkdir(parents=True, exist_ok=True)

    # Define the final PDF file path
    fpath = dest / f"{year}_{title}.pdf"

    print(f"[↓] {dtype} {year} {title}")

    try:
        # Prepare headers, copying the session headers
        headers = session.headers.copy()

        # 🔥 Critical for BSE/NSE: Set Referer header for valid downloading
        if "bseindia.com" in url:
            headers["Referer"] = "https://www.bseindia.com/"
        elif "nseindia.com" in url:
            headers["Referer"] = "https://www.nseindia.com/"

        # Execute the download stream request
        r = session.get(url, timeout=60, stream=True, headers=headers)
        r.raise_for_status()

        # ❌ detect HTML instead of PDF: Skip saving if the returned content is an HTML page
        content_type = r.headers.get("Content-Type", "")
        if "html" in content_type.lower():
            print("    ⚠ skipped (HTML page)")
            return False

        # Save the streamed chunks to the file
        with open(fpath, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)

        # Verify the downloaded file size
        size_kb = fpath.stat().st_size // 1024

        # Delete the file and fail if it's suspiciously small (under 10 KB)
        if size_kb < 10:
            fpath.unlink(missing_ok=True)
            print("    ⚠ skipped (too small)")
            return False

        print(f"    ✓ saved ({size_kb} KB)")
        return True

    except Exception as e:
        # Catch and log any download errors
        print(f"    ✗ failed: {e}")
        return False


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    # Setup command line argument parsing for the stock symbol
    ap = argparse.ArgumentParser()
    ap.add_argument("symbol")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    symbol = args.symbol.upper()

    # Initialize a requests session with default headers
    session = requests.Session()
    session.headers.update(HEADERS)

    print("\n[1] Fetching page")
    # Fetch the HTML and actual URL for the given symbol
    soup, page_url = fetch_page(session, symbol)

    print("[2] Extracting documents")
    # Parse the documents from the HTML soup
    docs = extract_documents(soup, page_url)

    print(f"✔ Clean docs found: {len(docs)}\n")

    # Display the found documents
    for d in docs:
        print(f"{d['doc_type']} | {d['year']} | {d['title']}")

    # Exit early if dry-run mode is enabled
    if args.dry_run:
        return

    # Define the output directory based on the symbol
    out_dir = Path("./screener_docs") / symbol

    print("\n[3] Downloading...\n")

    success = 0
    fail = 0

    # Note: Downloading only a slice for testing (docs[2:3])
    for doc in docs[2:3]:
        # Attempt to download each document and track success/failures
        if download_pdf(session, doc, out_dir):
            success += 1
        else:
            fail += 1
        # Add a short delay to avoid overwhelming the server
        time.sleep(1.5)

    # Print final summary report
    print("\n==============================")
    print(f"Downloaded: {success}")
    print(f"Failed: {fail}")
    print("==============================\n")


if __name__ == "__main__":
    main()

#python screener_downloader.py ADANIPORTS         <- run cmd 