#!/usr/bin/env python3
"""
LDA Topic Modeling — Privacy Requirements Engineering for AI Training Data
==========================================================================
Runs Latent Dirichlet Allocation (LDA) on a privacy-requirements corpus and
saves several publication-quality graphics plus an interactive HTML report to
the /app/output directory (mounted from the host).

Environment variables
---------------------
NUM_TOPICS  : number of LDA topics to extract (default: 20)
DATA_PATH   : path to the CSV dataset (default: /app/data/documents.csv)
OUTPUT_DIR  : directory where graphics are saved    (default: /app/output)
"""

import os
import re
import logging
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import gensim
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# ── silence non-critical warnings ────────────────────────────────────────────
warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
NUM_TOPICS  = int(os.getenv("NUM_TOPICS", 20))
DATA_PATH   = Path(os.getenv("DATA_PATH",  "/app/data/documents.csv"))
OUTPUT_DIR  = Path(os.getenv("OUTPUT_DIR", "/app/output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── seaborn / matplotlib style ────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PALETTE = sns.color_palette("tab10", NUM_TOPICS)

# ── NLTK assets ──────────────────────────────────────────────────────────────
for pkg in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words("english"))
# domain-specific additions
STOP_WORDS.update({
    "must", "shall", "should", "may", "data", "ai", "training",
    "personal", "system", "use", "used", "using", "include",
    "ensure", "provide", "process", "processing", "applied",
    "conducted", "required", "established", "implemented", "made",
    "based", "related", "also", "across", "upon", "within", "one",
    "least", "two", "three", "four", "five", "six", "seven",
    "defined", "documented", "identified", "reviewed", "maintained",
    "individual", "individuals", "organization", "organizations",
    "specific", "relevant", "appropriate", "available",
})

lemmatizer = WordNetLemmatizer()


# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_data(path: Path) -> pd.DataFrame:
    log.info(f"Loading dataset from {path}")
    df = pd.read_csv(path)
    log.info(f"  → {len(df)} records loaded")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PRE-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t.isalpha() and t not in STOP_WORDS and len(t) > 2]
    return tokens


def build_corpus(docs: list[list[str]]):
    dictionary = corpora.Dictionary(docs)
    dictionary.filter_extremes(no_below=2, no_above=0.95)
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    log.info(f"  Dictionary: {len(dictionary)} terms | Corpus: {len(corpus)} docs")
    return dictionary, corpus


# ─────────────────────────────────────────────────────────────────────────────
# 3.  LDA MODEL
# ─────────────────────────────────────────────────────────────────────────────
def train_lda(corpus, dictionary, num_topics: int) -> LdaModel:
    log.info(f"Training LDA with {num_topics} topics …")
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        update_every=1,
        chunksize=50,
        passes=20,
        alpha="auto",
        eta="auto",
    )
    log.info("  Training complete.")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 4.  GRAPHICS
# ─────────────────────────────────────────────────────────────────────────────

# ── 4a.  Top-words bar-chart per topic ───────────────────────────────────────
def plot_topic_word_barplot(model: LdaModel, num_topics: int) -> None:
    n_cols = min(num_topics, 3)
    n_rows = (num_topics + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(6 * n_cols, 4.5 * n_rows),
                             constrained_layout=True)
    axes = np.array(axes).flatten()
    fig.suptitle("LDA — Top 10 Words per Topic", fontsize=18, fontweight="bold", y=1.02)

    for idx in range(num_topics):
        top_words = model.show_topic(idx, topn=10)
        words  = [w for w, _ in reversed(top_words)]
        scores = [s for _, s in reversed(top_words)]
        ax = axes[idx]
        bars = ax.barh(words, scores, color=PALETTE[idx], edgecolor="white", linewidth=0.5)
        ax.set_title(f"Topic {idx + 1}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Probability weight", fontsize=10)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
        ax.set_xlim(0, max(scores) * 1.25)
        sns.despine(ax=ax, left=True)

    # hide unused subplot panels
    for i in range(num_topics, len(axes)):
        axes[i].set_visible(False)

    out = OUTPUT_DIR / "topic_word_barplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


# ── 4b.  Word clouds per topic ────────────────────────────────────────────────
def plot_wordclouds(model: LdaModel, num_topics: int) -> None:
    for idx in range(num_topics):
        freq = dict(model.show_topic(idx, topn=40))
        wc = WordCloud(
            width=800, height=450,
            background_color="white",
            colormap="viridis",
            max_words=40,
            prefer_horizontal=0.9,
        ).generate_from_frequencies(freq)

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"Topic {idx + 1} — Word Cloud", fontsize=16, fontweight="bold", pad=12)
        fig.patch.set_facecolor("#f7f7f7")

        out = OUTPUT_DIR / f"topic_wordcloud_{idx}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved → {out}")


# ── 4c.  Document × Topic distribution heatmap ───────────────────────────────
def plot_doc_topic_heatmap(model: LdaModel, corpus, num_topics: int) -> None:
    doc_topics = np.zeros((len(corpus), num_topics))
    for i, bow in enumerate(corpus):
        for tid, prob in model.get_document_topics(bow, minimum_probability=0.0):
            doc_topics[i, tid] = prob

    sample_size = min(40, len(corpus))
    sample_idx  = np.random.choice(len(corpus), sample_size, replace=False)
    sample_idx.sort()
    matrix = doc_topics[sample_idx]

    fig, ax = plt.subplots(figsize=(max(8, num_topics * 1.2), sample_size * 0.35 + 2))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        xticklabels=[f"T{i+1}" for i in range(num_topics)],
        yticklabels=[f"Doc {i+1}" for i in sample_idx],
        linewidths=0.3,
        linecolor="white",
        annot=(num_topics <= 20),
        fmt=".2f",
        cbar_kws={"label": "Topic Probability"},
    )
    ax.set_title("Document × Topic Probability Distribution", fontsize=15, fontweight="bold", pad=14)
    ax.set_xlabel("Topic", fontsize=12)
    ax.set_ylabel("Document", fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0, fontsize=7)

    out = OUTPUT_DIR / "document_topic_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


# ── 4d.  Coherence score vs. number of topics ────────────────────────────────
def plot_coherence_curve(tokenized: list[list[str]], dictionary, corpus) -> None:
    k_range   = range(2, min(12, len(corpus) // 5 + 2))
    coherence_vals = []
    log.info("Computing coherence scores …")
    for k in k_range:
        m = LdaModel(corpus=corpus, id2word=dictionary,
                     num_topics=k, random_state=42,
                     passes=10, chunksize=50)
        cm = CoherenceModel(model=m, texts=tokenized,
                            dictionary=dictionary, coherence="c_v")
        coherence_vals.append(cm.get_coherence())
        log.info(f"    k={k:2d}  coherence={coherence_vals[-1]:.4f}")

    best_k = list(k_range)[np.argmax(coherence_vals)]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(list(k_range), coherence_vals, "o-", color="#4C72B0",
            linewidth=2.5, markersize=7, markerfacecolor="white",
            markeredgewidth=2, label="Coherence (c_v)")
    ax.axvline(best_k, linestyle="--", color="#DD8452",
               linewidth=1.8, label=f"Best k = {best_k}")
    ax.fill_between(list(k_range), coherence_vals,
                    alpha=0.12, color="#4C72B0")
    ax.set_xlabel("Number of Topics (k)", fontsize=13)
    ax.set_ylabel("Coherence Score (c_v)", fontsize=13)
    ax.set_title("Optimal Number of Topics — Coherence Curve", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(list(k_range))
    sns.despine(ax=ax)

    out = OUTPUT_DIR / "topic_coherence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}  (best k={best_k})")


# ── 4e.  Topic weight stacked bar chart (corpus summary) ─────────────────────
def plot_topic_prevalence(model: LdaModel, corpus, num_topics: int) -> None:
    totals = np.zeros(num_topics)
    for bow in corpus:
        for tid, prob in model.get_document_topics(bow, minimum_probability=0.0):
            totals[tid] += prob
    totals /= len(corpus)

    labels = [f"Topic {i+1}" for i in range(num_topics)]
    colors = [PALETTE[i] for i in range(num_topics)]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, totals, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    ax.set_xlabel("Topic", fontsize=13)
    ax.set_ylabel("Average Topic Proportion", fontsize=13)
    ax.set_title("Topic Prevalence Across Corpus", fontsize=15, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
    ax.set_ylim(0, max(totals) * 1.25)
    sns.despine(ax=ax)

    out = OUTPUT_DIR / "topic_prevalence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved → {out}")


# ── 4f.  pyLDAvis interactive HTML ───────────────────────────────────────────
def save_pyldavis(model: LdaModel, corpus, dictionary) -> None:
    log.info("Generating pyLDAvis interactive visualization …")
    vis = gensimvis.prepare(model, corpus, dictionary, sort_topics=False)
    out = OUTPUT_DIR / "lda_visualization.html"
    pyLDAvis.save_html(vis, str(out))
    log.info(f"  Saved → {out}")


# ── 4g.  Export Document-Topic Probabilities for Labeler ─────────────────────
def export_doc_topic_probs(model: LdaModel, corpus, df: pd.DataFrame, num_topics: int) -> None:
    import json
    log.info("Exporting document-topic probabilities to JavaScript file …")
    
    doc_topics = np.zeros((len(corpus), num_topics))
    for i, bow in enumerate(corpus):
        for tid, prob in model.get_document_topics(bow, minimum_probability=0.0):
            doc_topics[i, tid] = prob
            
    data = []
    for i in range(len(df)):
        data.append({
            "id": i + 1,
            "text": str(df["text"].iloc[i]),
            "probs": [round(p, 4) for p in doc_topics[i].tolist()]
        })
        
    out = OUTPUT_DIR / "doc_topic_probs.js"
    with open(out, "w", encoding="utf-8") as f:
        f.write("const DOC_DATA = ")
        json.dump(data, f)
        f.write(";\n")
    log.info(f"  Saved → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  REPORT (terminal summary)
# ─────────────────────────────────────────────────────────────────────────────
def print_topic_summary(model: LdaModel, num_topics: int) -> None:
    print("\n" + "=" * 65)
    print(f"  LDA TOPIC SUMMARY  ({num_topics} topics)")
    print("=" * 65)
    for idx in range(num_topics):
        top_words = ", ".join(w for w, _ in model.show_topic(idx, topn=8))
        print(f"  Topic {idx + 1:2d}: {top_words}")
    print("=" * 65 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("=" * 60)
    log.info("  LDA Topic Modeling — Privacy Requirements")
    log.info("=" * 60)

    # 1. Load
    df = load_data(DATA_PATH)

    # 2. Pre-process
    log.info("Pre-processing text …")
    tokenized = [preprocess(t) for t in df["text"]]

    # 3. Build corpus
    log.info("Building corpus …")
    dictionary, corpus = build_corpus(tokenized)

    # 4. Train LDA
    model = train_lda(corpus, dictionary, NUM_TOPICS)

    # 5. Print summary
    print_topic_summary(model, NUM_TOPICS)

    # 6. Generate graphics
    log.info("Generating graphics …")

    log.info("  [1/6] Topic word bar-chart")
    plot_topic_word_barplot(model, NUM_TOPICS)

    log.info("  [2/6] Word clouds")
    plot_wordclouds(model, NUM_TOPICS)

    log.info("  [3/6] Document-topic heatmap")
    plot_doc_topic_heatmap(model, corpus, NUM_TOPICS)

    log.info("  [4/6] Coherence curve")
    plot_coherence_curve(tokenized, dictionary, corpus)

    log.info("  [5/6] Topic prevalence bar-chart")
    plot_topic_prevalence(model, corpus, NUM_TOPICS)

    log.info("  [6/7] pyLDAvis HTML")
    save_pyldavis(model, corpus, dictionary)

    log.info("  [7/7] Document Probabilities JS")
    export_doc_topic_probs(model, corpus, df, NUM_TOPICS)

    log.info("")
    log.info("All outputs saved to: %s", OUTPUT_DIR)
    log.info("Done ✓")


if __name__ == "__main__":
    main()
