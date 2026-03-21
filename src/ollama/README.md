# Ollama Service

This directory contains the configuration and setup for the Ollama service, which provides local Large Language Model (LLM) inference capabilities for the dissertation experiment project.

## Overview

Ollama is a lightweight, self-hosted solution for running large language models locally. In this project, it serves as the LLM backend for various AI-powered components, including the CrewAI agents used in the document processing pipeline.

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

The Ollama service integrates with other components in the project:

1. **CrewAI Agents**: Uses Ollama as the LLM provider for intelligent document processing and analysis
2. **Vector Pipeline**: Provides semantic understanding capabilities for text processing
3. **API Layer**: Accessible via the `appVectorApi` service for LLM inference requests

## Configuration

The service is configured through Docker Compose and uses:

- **Port**: Typically exposed on port 11434
- **Volume Mounts**: Persistent storage for model files
- **Network**: Connected to the project's internal Docker network for inter-service communication

## Usage

The Ollama service is automatically started as part of the Docker Compose stack. It downloads and runs language models on-demand based on the requests from the CrewAI agents and other components.

### Model Management

Models are managed automatically by the Ollama service. Common models used in this project may include:

- Llama 2/3 for general-purpose text generation
- Mistral for efficient inference
- Code-specific models for technical documentation processing

## Technical Details

- **Runtime**: Runs as a Docker container within the project's microservices architecture
- **Dependencies**: No external API keys required - fully self-hosted
- **Performance**: Leverages available GPU resources when available, falls back to CPU inference
