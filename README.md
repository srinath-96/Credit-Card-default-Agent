# Credit Risk Analysis Chatbot

This project is an intelligent chatbot designed to assist with credit risk analysis. Built using the `smolagents` library, it leverages a `CodeAgent` to seamlessly interpret user queries and execute relevant tools (e.g., data visualization, modeling, research retrieval) on a provided dataset. The interface is powered by a customized Gradio UI, allowing for interactive chat and file uploads with support for additional file types like `.csv`.

## Features

- **Natural Language Interaction**: Ask questions like "What’s in the dataset?" or "Run a model" and get conversational responses.
- **Tool Integration**: Automatically selects and runs tools such as:
  - Data visualization (`DataVisualizationTool`)
  - Machine learning modeling (`ModelingTool`, capped at 3 steps)
  - Research paper retrieval (`RetrieverTool`)
  - Web search (`DuckDuckGoSearchTool`)
- **Custom Gradio UI**: Interactive web interface with support for file uploads (e.g., `.pdf`, `.csv`).
- **Flexible File Types**: Extends default GradioUI to allow additional file formats beyond `.pdf`, `.docx`, and `.txt`.

## Prerequisites

- **Python**: 3.8 or higher
- **Dependencies**:
  - `smolagents[gradio]`: Install with `pip install 'smolagents[gradio]'`
  - `pandas`: For dataset handling
  - Your custom tools (e.g., `ModelingTool`, `DataVisualizationTool`, etc.)



