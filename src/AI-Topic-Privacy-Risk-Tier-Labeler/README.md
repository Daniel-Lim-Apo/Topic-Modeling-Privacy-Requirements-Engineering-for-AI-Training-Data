# AI-Topic-Privacy-Risk-Tier-Labeler

This component operationalizes the **Evaluate** and **Score** phases of the Privacy-Aware Requirements Engineering methodology. 

## Overview
Once the Latent Dirichlet Allocation (LDA) component extracts the topics and the document-topic distribution matrix from the raw textual corpus, this labeler component steps in to process that data. 

It executes the following tasks:
1. **Topic Evaluation**: It ingests the top keywords for each topic identified by the LDA model and queries the local LLM (via the `ollama` service). The LLM acts as an AI agent to assign a concise name, a short description, and a **privacy risk tier** (on a scale from 1 = Minimal Risk to 5 = Critical Risk) to each topic based on the sensitivity of the keywords.
2. **Document Risk Scoring**: Using the evaluated privacy risk tiers for each topic and the proportion of each topic present in every document, it calculates a final accumulated Privacy Risk Score for each document.
3. **Output Generation**: It produces an updated dataset (`document_risks_ai.csv`) where documents are ranked by their privacy risk score. This acts as the key artifact to support Requirements Engineering (RE) decision-making.

## Setup & Execution
This service is designed to be run as part of the overall Docker Compose stack. 

### Dependencies
- The `LDA` service must have completed its run and generated the `doc_topic_probs.js` output.
- The `ollama` service must be running and accessible over the network to provide LLM inference.

### Environment Variables
- `OUTPUT_DIR`: The directory where the component will look for LDA outputs and save its own outputs.
- `OLLAMA_URL`: The API endpoint for the Ollama service (defaults to `http://host.docker.internal:11434/api/generate`).
- `OLLAMA_MODEL`: The LLM model to be used for the evaluation (e.g., `llama3`).
