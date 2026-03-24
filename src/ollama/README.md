# Ollama Service

This directory contains the configuration and setup for the Ollama service, which provides local Large Language Model (LLM) inference capabilities for the dissertation experiment project.

## Overview

Ollama is a lightweight, self-hosted solution for running large language models locally. In this methodology's **Evaluate** phase, it serves as the LLM backend acting as the AI agent. It is responsible for analyzing topic keywords and assigning semantic names, descriptions, and privacy risk tiers.

## Components

### Dockerfile
The `Dockerfile` extends the official `ollama/ollama` base image with project-specific configurations:
- **Base Image**: `ollama/ollama` - Official Ollama container image
- **Timezone**: Set to `America/Sao_Paulo` for consistent timestamp handling
- **Certificate Management**: Configured to support custom CA certificates for corporate network environments

### start.ollama.sh
A simple shell script that:
- Initializes the Ollama server
- Starts the Ollama service in server mode
- Provides startup logging for monitoring

## Integration

The Ollama service closely integrates with the **AI-Topic-Privacy-Risk-Tier-Labeler** component. By exposing a local REST API, it allows the labeler to submit prompts containing LDA-extracted keywords and receive structured JSON responses characterizing the privacy risks of each topic.

## Configuration

The service is configured through Docker Compose and uses:
- **Port**: Typically exposed on port 11434
- **Volume Mounts**: Persistent storage for downloaded model weights
- **Network**: Connected to the project's internal Docker network for inter-service communication

## Usage

The Ollama service is automatically started as part of the Docker Compose stack. It manages the lifecycle of models automatically based on requests. 

When the `AI-Topic-Privacy-Risk-Tier-Labeler` initializes, it ensures the required model (e.g. `llama3`) is pulled and ready for inference before querying the API.

## Technical Details

- **Runtime**: Runs as a Docker container within the project's microservices architecture
- **Dependencies**: No external API keys required - fully self-hosted, keeping sensitive data analysis entirely local
- **Performance**: Leverages available GPU resources when available; falls back to CPU inference if necessary
