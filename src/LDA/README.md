# LDA Topic Modeling — Privacy Requirements for AI Training Data

This component operationalizes the **Preprocess** and **Train** phases of the Privacy-Aware Requirements Engineering methodology. 

## Overview
This module applies **Latent Dirichlet Allocation (LDA)** to a corpus of raw textual data to extract latent semantic structures. By representing documents as mixtures of topics and topics as distributions of words, it sets the foundation for downstream privacy assessment.

The pipeline performs text preprocessing and utilizes the `gensim` framework to generate the topic models. It outputs the document-topic distribution weights and top topic keywords required by the `AI-Topic-Privacy-Risk-Tier-Labeler`, alongside several publication-quality graphics for manual review.

---

## Project Structure

```
src/LDA/
├── data/
│   └── documents.csv   # 200-sentence privacy-requirements corpus
├── src/
│   └── lda_analysis.py            # end-to-end LDA pipeline
├── output/                        # ← generated artifacts and models appear here
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Quick Start
*Note: It is recommended to run this as part of the main `docker-compose.yml` at the project root.*

```bash
# 1 — build the Docker image
docker compose build

# 2 — run the analysis (outputs saved to ./output/)
docker compose up

# 3 — view interactive results
open output/lda_visualization.html   # interactive pyLDAvis in browser
```

---

## Output Artifacts

| File | Description |
|------|-------------|
| `doc_topic_probs.js` | Topic distribution matrix consumed by the Labeler Component |
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
| `OUTPUT_DIR` | `/app/output` | Where graphics and data are written |

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
