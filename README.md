# 📝 AI Text Summarizer with DistilBART

An interactive web application built with Gradio and Hugging Face Transformers that leverages a distilled sequence-to-sequence model to instantly generate abstractive summaries from long paragraphs or articles.

---

## 🚀 Features
* **Abstractive Summarization:** Generates brand new, coherent sentences to summarize information rather than just copying and cutting existing text lines.
* **Speed-Optimized Architecture:** Powered by `distilbart-cnn-12-6`, which uses a full 12-layer encoder for understanding text but splits the decoder to 6 layers for faster token generation.
* **Hardware Adaptive:** Automatically detects and pairs with an **NVIDIA GPU (CUDA)** for fast inference processing, falling back safely to standard CPU if running in cloud spaces.
* **Web User Interface:** A clean, easy-to-use textbox dashboard built entirely inside the browser using Gradio.

