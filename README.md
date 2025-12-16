# Virality-Website
# Sentiment → Virality Predictor

This project explores how sentiment influences content virality by combining
a fine-tuned BERT sentiment classifier with an SEIR-based diffusion model.

## Overview
- A BERT model predicts sentiment from Instagram captions or news-style headlines
- The predicted sentiment conditions an SEIR model to simulate virality over time
- Results are visualized through a graph

## Files
- `index.html` — Frontend interface (deployable via GitHub Pages)
- `server.py` — FastAPI backend that loads the BERT model and runs the SEIR simulation

## Model Weights
Due to size constraints, the fine tuned BERT model is not in this repository.
The model was trained locally on BBC News caption data.
