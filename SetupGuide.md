# Financial RAG Setup Guide

This guide explains how to set up the Financial RAG ingestion pipeline from scratch before running `ingest.py`.

---

# 1. System Requirements

- Python 3.10 or newer
- Docker Desktop
- Git
- Windows/Linux/macOS

Recommended:

- 16 GB RAM
- SSD storage
- Internet connection for downloading embedding models

---

# 2. Clone the Repository

```bash
git clone <repository-url>
cd Financial-Rag
```

---

# 3. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# 4. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

# 5. Configure Environment Variables

Create a `.env` file in the project root.

Example:

```env
# -------------------------
# MinIO
# -------------------------
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=False

# -------------------------
# MySQL
# -------------------------
MYSQL_HOST=localhost
MYSQL_PORT=3307
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=ai_hedge_fund

# -------------------------
# Qdrant
# -------------------------
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# -------------------------
# Embedding Model
# -------------------------
EMBEDDING_MODEL=FinLang/finance-embeddings-investopedia
EMBEDDING_DIM=768

# -------------------------
# Logging
# -------------------------
LOG_DIR=logs
```

> **Important:** `QDRANT_HOST` should only contain the hostname (for example `localhost`). Do not include `http://` or `https://`.

---

# 6. Start Required Services

The ingestion pipeline depends on three services.

| Service | Purpose | Default Port |
|----------|---------|--------------|
| MinIO | PDF Storage | 9000 |
| MinIO Console | Management UI | 9001 |
| MySQL | Metadata Database | 3307 |
| Qdrant | Vector Database | 6333 |

If using Docker:

```bash
docker start finrag_qdrant
docker start finrag_minio
docker start finrag_mysql
```

Verify the containers are running:

```bash
docker ps
```

---

# 7. Verify Qdrant

Run:

```bash
curl http://localhost:6333
```

Expected response:

```json
{
  "title": "qdrant"
}
```

---

# 8. Verify MinIO

Open the MinIO Console:

```
http://localhost:9001
```

Log in using your configured credentials.

Create the following buckets if they do not already exist:

```
annual-reports
concall-transcripts
```

---

# 9. Upload PDFs

Store files using the following folder structure.

Annual reports:

```
annual-reports/
    hal/
        2019_Annual_Report.pdf
        2020_Annual_Report.pdf
```

Concall transcripts:

```
concall-transcripts/
    hal/
        2024_Q4_Transcript.pdf
```

The company folder name should be the stock symbol in lowercase.

---

# 10. Filename Convention

The ingestion pipeline extracts the year from the first four digits of the filename.

Correct examples:

```
2024_Annual_Report.pdf
2023_Q4_Transcript.pdf
2018_Report.pdf
```

Incorrect example:

```
Annual_Report_2024.pdf
```

If the filename does not begin with a year, the document is still ingested, but its `year` field will be stored as `NULL`.

---

# 11. Verify MySQL

Confirm that MySQL is running.

Example:

```bash
mysql -h localhost -P 3307 -u root -p
```

The database should exist:

```
ai_hedge_fund
```

The application automatically creates the required tables during initialization.

---

# 12. Download Embedding Model

The first execution downloads the configured embedding model from Hugging Face.

Default model:

```
FinLang/finance-embeddings-investopedia
```

This happens only once and may take several minutes depending on internet speed.

---

# 13. Project Directory

A typical directory structure looks like:

```
Financial-Rag/
│
├── config/
├── loaders/
├── extractors/
├── chunkers/
├── embeddings/
├── utils/
├── logs/
├── ingest.py
├── requirements.txt
├── .env
└── README.md
```

---

# 14. Verify Everything

Before running ingestion, confirm:

- Virtual environment is activated.
- Dependencies are installed.
- Docker containers are running.
- MinIO buckets exist.
- PDFs are uploaded.
- `.env` configuration is correct.
- Qdrant responds on port 6333.
- MySQL is accessible.

---

# 15. Ready for Ingestion

Once the setup is complete, the ingestion pipeline can be executed.

Examples:

```bash
python ingest.py --list
```

```bash
python ingest.py --symbol HAL
```

```bash
python ingest.py --all
```

```bash
python ingest.py --stats
```

For detailed ingestion options and troubleshooting, refer to the separate **Running `ingest.py` Guide**.