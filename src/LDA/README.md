# LDA Topic Modeling — Privacy Requirements for AI Training Data

> Applies **Latent Dirichlet Allocation (LDA)** to a corpus of privacy-requirements sentences and generates several publication-quality graphics.

---

## Project Structure

```
src/LDA/
├── data/
│   └── documents.csv   # 200-sentence privacy-requirements corpus
├── src/
│   └── lda_analysis.py            # end-to-end LDA pipeline
├── output/                        # ← graphics appear here after running
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Quick Start

```bash
# 1 — build the Docker image
docker compose build

# 2 — run the analysis (outputs saved to ./output/)
docker compose up

# 3 — view results
open output/topic_word_barplot.png
open output/lda_visualization.html   # interactive pyLDAvis in browser
```

---

## Output Graphics

| File | Description |
|------|-------------|
| `topic_word_barplot.png` | Top-10 words per topic (bar charts) |
| `topic_wordcloud_N.png` | Word cloud for each topic N |
| `document_topic_distribution.png` | Heatmap: documents × topic weights |
| `topic_coherence.png` | Coherence score vs. number of topics |
| `topic_prevalence.png` | Average topic proportion across corpus |
| `lda_visualization.html` | Interactive pyLDAvis explorer |

---

## Configuration

Override defaults with environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_TOPICS` | `20` | Number of LDA topics |
| `DATA_PATH` | `/app/data/documents.csv` | Input CSV |
| `OUTPUT_DIR` | `/app/output` | Where graphics are written |

Example — change to 7 topics:
```yaml
environment:
  - NUM_TOPICS=7
```

---

## Dataset

`data/documents.csv` contains **200 curated sentences** covering:
- Consent & legal basis
- Anonymization & pseudonymization
- Data minimization & retention
- Security & access control
- Transparency & accountability
- Data subject rights (access, erasure, portability)
- Privacy by design & breach response
- Cross-border transfers & DPIA

The dataset is based on typical privacy-engineering requirement themes aligned with GDPR and AI governance frameworks.
