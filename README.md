# 🧠 Explainable AI Toolkit for Multimodal Models

## Overview
This project presents an **Explainable AI (XAI) toolkit** designed to address the *black-box problem* in multimodal language models.  
It enables users to **extract, aggregate, and visualize attention mechanisms** from the [`llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf) model.

The toolkit provides an interpretable view of the model’s internal reasoning by generating **attention heatmaps** that illustrate the relationship between generated text tokens and visual input regions — effectively answering the question:  
> *Where did the model look when generating this token?*

Our pipeline supports:
- Extraction of attention weights across **specific steps, layers and heads**
- **Aggregation** of multi-head attention patterns for clearer interpretation
- **Visualization** of attention as overlayed heatmaps on input images

This toolkit contributes to enhancing transparency and interpretability in multimodal generative models, allowing researchers to analyze how multimodal language models integrate visual and textual information during generation.
