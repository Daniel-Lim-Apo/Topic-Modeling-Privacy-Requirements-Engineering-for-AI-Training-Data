import os
import json
import re
import csv
import logging
import requests
import pandas as pd
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

# Constants
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "/app/output")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

DOC_TOPIC_PROBS_PATH = os.path.join(OUTPUT_DIR, "doc_topic_probs.js")
DOCUMENT_RISKS_PATH = os.path.join(OUTPUT_DIR, "document_risks.csv")
TOPIC_LABELS_AI_PATH = os.path.join(OUTPUT_DIR, "topic_labels_ai.json")
DOCUMENT_RISKS_AI_PATH = os.path.join(OUTPUT_DIR, "document_risks_ai.csv")

def extract_js_data(filepath):
    log.info(f"Extracting data from {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract DOC_DATA
    doc_data_match = re.search(r'const DOC_DATA = (\[.*?\]);', content, re.DOTALL)
    doc_data = json.loads(doc_data_match.group(1)) if doc_data_match else []
    
    # Extract TOP_KEYWORDS
    top_keywords_match = re.search(r'const TOP_KEYWORDS = (\[.*?\]);', content, re.DOTALL)
    top_keywords = json.loads(top_keywords_match.group(1)) if top_keywords_match else []
    
    return doc_data, top_keywords

def query_ollama(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json"
    }
    log.info(f"Querying Ollama at {OLLAMA_URL} with model {OLLAMA_MODEL}")
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json().get('response', '{}')
        
        # Try finding json bracket match to avoid markdown conversational wrapper
        json_match = re.search(r'\{.*\}', result, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
            
        return json.loads(result)
    except Exception as e:
        log.error(f"Error querying Ollama: {e}")
        # Return fallback json in case of failure
        return {
            "name": "Unknown Topic",
            "description": "Failed to generate description",
            "privacy_risk_tier": 0,
            "privacy_risk_label": "Unknown"
        }

def label_topics(top_keywords):
    topics = []
    for i, keywords in enumerate(top_keywords):
        topic_number = i + 1
        log.info(f"Labeling Topic {topic_number} out of {len(top_keywords)}")
        
        prompt = f"""
You are an expert in privacy requirements engineering and AI data protection.
Given the following top keywords extracted from a single topic in a privacy-requirements corpus:
{', '.join(keywords)}

Based on these keywords:
1. Provide a concise, descriptive 'name' for this topic (e.g. 'Data Security and Protection', 'User Consent and Rights').
2. Provide a short 'description' of what this topic entails (1-2 sentences).
3. Assign a 'privacy_risk_tier' on a scale of 1 to 5, where 1 is Minimal Risk and 5 is Critical Risk.
4. Assign a 'privacy_risk_label' corresponding to the tier (e.g. Minimal, Low, Moderate, High, Critical).

Output exactly and only a valid JSON object matching this structure:
{{
  "name": "Topic Name",
  "description": "Topic Description",
  "privacy_risk_tier": 2,
  "privacy_risk_label": "Low"
}}
"""
        result_json = query_ollama(prompt)
        
        topic_info = {
            "topic_number": topic_number,
            "name": result_json.get("name", f"Topic {topic_number}"),
            "description": result_json.get("description", ""),
            "privacy_risk_tier": result_json.get("privacy_risk_tier", 0),
            "privacy_risk_label": result_json.get("privacy_risk_label", "Unknown"),
            "top_keywords": keywords
        }
        topics.append(topic_info)
        
    return topics

def calculate_new_risk_scores(document_risks_path, doc_data, topics, output_path):
    log.info(f"Calculating new risk scores based on AI-generated tiers")
    
    # Create an array of tiers (0-indexed to match topic format: topic 1 is at index 0)
    tiers = [t["privacy_risk_tier"] for t in topics]
    
    try:
        df = pd.read_csv(document_risks_path)
        # Strip stray leading/trailing quotes and spaces from column names
        # (the source CSV has a leading ' on the header row)
        df.columns = df.columns.str.strip().str.strip("'\"")
        log.info(f"Loaded {len(df)} documents from {document_risks_path}. Columns: {list(df.columns)}")
    except Exception as e:
        log.error(f"Could not load {document_risks_path}: {e}")
        return
    
    # Build a map: sequential_position -> Risk_Score
    # doc["id"] in doc_topic_probs.js is a sequential 1-based index that corresponds
    # to the original row order in the source dataframe that LDA was trained on.
    # Document_ID in document_risks.csv is the original row index from the source dataset
    # (not sequential 1..N, but the actual dataset row numbers like 390, 711, etc.).
    # Sorting df by Document_ID restores the original processing order so we can
    # match position i (0-based) to doc["id"] = i+1 (1-based).
    doc_id_to_score = {}
    for doc in doc_data:
        probs = doc["probs"]
        score = sum(p * t for p, t in zip(probs, tiers))
        doc_id_to_score[doc["id"]] = round(score, 4)
    
    log.info(f"Computed {len(doc_id_to_score)} risk scores from doc_topic_probs.js")
    
    # Sort df by Document_ID ascending to restore original processing order
    df_sorted = df.sort_values(by="Document_ID").reset_index(drop=True)
    df_sorted["Risk_Score"] = df_sorted.index.map(
        lambda i: doc_id_to_score.get(i + 1, 0.0)
    )
    
    # Check how many documents got a non-zero score
    matched = (df_sorted["Risk_Score"] > 0).sum()
    log.info(f"Matched {matched} out of {len(df_sorted)} documents with non-zero risk scores")

    # Sort by Risk_Score descending for output
    df_out = df_sorted.sort_values(by="Risk_Score", ascending=False)
    
    try:
        df_out.to_csv(output_path, index=False)
        log.info(f"Saved updated document risks to {output_path}")
    except Exception as e:
        log.error(f"Failed to save {output_path}: {e}")

def ensure_model_pulled(model_name):
    pull_url = OLLAMA_URL.replace('/api/generate', '/api/pull')
    log.info(f"Ensuring model '{model_name}' is available. This may take a few minutes if it needs to download...")
    try:
        response = requests.post(pull_url, json={"name": model_name, "stream": False}, timeout=None)
        response.raise_for_status()
        log.info(f"Model '{model_name}' is ready.")
    except Exception as e:
        log.error(f"Failed to pull model {model_name}: {e}")

def main():
    log.info("Starting AI Topic Privacy Risk Tier Labeler")
    
    if not os.path.exists(DOC_TOPIC_PROBS_PATH):
        log.error(f"Error: {DOC_TOPIC_PROBS_PATH} not found.")
        return
        
    doc_data, top_keywords = extract_js_data(DOC_TOPIC_PROBS_PATH)
    
    if not top_keywords:
        log.error("Error: Could not extract TOP_KEYWORDS from JS file.")
        return
        
    log.info(f"Successfully extracted {len(doc_data)} document records and {len(top_keywords)} topics.")
    
    ensure_model_pulled(OLLAMA_MODEL)
    
    # Label topics using Ollama
    topics = label_topics(top_keywords)
    
    # Save the AI generated topic labels
    output_json = {
        "meta": {
            "tool": "AI Topic Labeler",
            "saved_at": datetime.utcnow().isoformat() + "Z",
            "num_topics": len(topics)
        },
        "topics": topics
    }
    
    os.makedirs(os.path.dirname(TOPIC_LABELS_AI_PATH), exist_ok=True)
    with open(TOPIC_LABELS_AI_PATH, 'w') as f:
        json.dump(output_json, f, indent=2)
    log.info(f"Saved AI topic labels to {TOPIC_LABELS_AI_PATH}")
    
    # Update document risks
    if os.path.exists(DOCUMENT_RISKS_PATH):
        calculate_new_risk_scores(DOCUMENT_RISKS_PATH, doc_data, topics, DOCUMENT_RISKS_AI_PATH)
    else:
        log.warning(f"File not found: {DOCUMENT_RISKS_PATH}. Skipping risk score recalculation.")

if __name__ == "__main__":
    main()
