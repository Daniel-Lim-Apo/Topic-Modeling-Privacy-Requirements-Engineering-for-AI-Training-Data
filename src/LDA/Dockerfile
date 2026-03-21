# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System dependencies (matplotlib / Pillow / wordcloud) ─────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libfreetype6-dev \
        libpng-dev \
        libfontconfig1 \
        gcc \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Pre-download NLTK corpora at build time ───────────────────────────────────
RUN python -c "\
import nltk; \
nltk.download('punkt'); \
nltk.download('punkt_tab'); \
nltk.download('stopwords'); \
nltk.download('wordnet'); \
nltk.download('omw-1.4')"

# ── Copy application files ────────────────────────────────────────────────────
COPY data/ ./data/
COPY src/  ./src/

# ── Create output directory (will be overridden by volume mount) ──────────────
RUN mkdir -p /app/output

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["python", "src/lda_analysis.py"]
