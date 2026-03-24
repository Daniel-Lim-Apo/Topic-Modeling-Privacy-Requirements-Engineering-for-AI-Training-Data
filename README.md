# Topic-Modeling-Privacy-Requirements-Engineering-for-AI-Training-Data
Using Topic Modeling to Support Privacy-Aware Requirements Engineering for AI Training Data

This repository contains a proof-of-concept (PoC) implementation of the methodology described in the study *"A Semi-Automated Approach to Support Privacy-Aware Requirements Engineering for AI Training Data"*.

## Overview
Raw textual data used to train AI models may contain sensitive information, raising privacy risks and compliance concerns. Manual inspection of such data is costly and difficult to scale. From a Requirements Engineering (RE) perspective, this creates the need for systematic approaches to support the early identification and assessment of privacy-related concerns before textual data are incorporated into AI systems.

This solution proposes a semi-automated, Latent Dirichlet Allocation (LDA)-based approach to support privacy-aware RE by enabling the identification and classification of privacy risks in raw textual data used for AI training.

## Methodology Workflow
The approach is composed of four main phases:

1. **Preprocess**: Text collection, data cleaning, normalization, and filtering to construct a document-term matrix.
2. **Train (LDA Component)**: Extracts latent topics from textual corpora using LDA. The model outputs a probability distribution over the vocabulary for each topic, and a probability distribution over topics for each document.
3. **Evaluate (Labeler & Ollama Components)**: Extracted topics are interpreted through expert judgment, supported by human reviewers and LLM-based AI agents, to assign a name, description, and privacy risk tier (1-5) to each topic.
4. **Score (Labeler Component)**: Calculates the final privacy risk score for each document within the corpus based on the topic distributions and the evaluated privacy risk tiers of those topics. This informs RE decision-making regarding the suitability of the training data.

## Repository Components

This PoC comprises three main modular components, each containerized via Docker:

* [**LDA Component**](./src/LDA/README.md)
  Handles the *Preprocess* and *Train* phases of the methodology. It ingests the text corpus, applies text processing, executes the LDA topic modeling using the `gensim` framework, and visualizes the results.

* [**Ollama Service**](./src/ollama/README.md)
  Provides local Large Language Model (LLM) inference capabilities. It is utilized during the *Evaluate* phase to act as the AI agent that semi-automatically labels topics with privacy risk tiers and descriptions.

* [**AI-Topic-Privacy-Risk-Tier-Labeler**](./src/AI-Topic-Privacy-Risk-Tier-Labeler/README.md)
  Handles the *Evaluate* and *Score* phases. It consumes the LDA model outputs and interacts with the Ollama service to classify the topics. Then, it calculates the overall privacy risk score for each document to support RE validation activities.

## Getting Started
To run the full pipeline, instantiate the docker containers using the `docker-compose.yml` file located in the root directory. Refer to the specific README of each component for detailed instructions and configurations.
