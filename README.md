---
title: Ground Up Reels Bot
emoji: 🍳
colorFrom: yellow
colorTo: orange
sdk: docker
app_port: 7860
pinned: false
---

# Ground Up Reels Bot

A RAG chatbot for Instagram Reels content using Gemini and a local SQLite vector database.

## Deploying on Hugging Face Spaces

1. Create a new Space on [Hugging Face Spaces](https://huggingface.co/spaces).
2. Choose **Docker** as the SDK.
3. Link your GitHub repository to the Space or push this code to the Space's Git remote.
4. Add your **`GEMINI_API_KEY`** in the Space Settings under **Variables and Secrets** as a **Secret**.
5. Hugging Face will automatically build the `Dockerfile` and launch the Chainlit interface!
