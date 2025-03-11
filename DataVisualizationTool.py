from dotenv import load_dotenv
from typing import Any, Dict
from smolagents import CodeAgent, Tool,LiteLLMModel
import pandas as pd
import numpy as np
import os
from PyPDF2 import PdfReader
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
from ucimlrepo import fetch_ucirepo
import h2o

class DataVisualizationTool(Tool):
    name = "data_visualizer"
    description = "Creates insightful visualizations and analysis from dataframes. Provide the dataframe to analyze."
    inputs = {
        "dataset": {
            "type": "object",
            "description": "Dataframe to analyze and visualize"
        },
        "visualization_type": {
            "type": "string",
            "description": "Type of analysis to perform",
            "nullable": True
        }
    }
    output_type = "object"

    def forward(self, dataset, visualization_type=None):
        # Use the consistent prompt regardless of visualization_type
        analysis_prompt = """
        You are a data analysis expert tasked with analyzing and visualizing a dataset using pandas as the primary DataFrame for data handling, and Plotly and Matplotlib for visualizations. Do not use h2o, scipy, sklearn, seaborn, or any other libraries unless explicitly stated. Follow these precise steps:

        1. Load and examine the dataset:
           - Import pandas to load the dataset from a provided CSV file (e.g., 'dataset.csv') using pd.read_csv().
           - Store the dataset in a pandas DataFrame (e.g., df).
           - Print the first 5 rows using df.head() and the dataset shape using df.shape.
           - Describe the dataset in text:
             - Number of rows and columns.
             - Column names and their data types (use df.dtypes).
             - Explanation of components: Identify the target column 'Y' (binary: 0 or 1) and all other columns as features.

        2. Generate descriptive statistics:
           - Use pandas to compute statistics with df.describe() for numerical columns and df['Y'].value_counts() for the target column.
           - Print the results clearly, labeling them as 'Numerical Statistics' and 'Target Class Distribution'.

        3. Create relevant visualizations using Plotly and Matplotlib:
           - Understand that 'Y' is the target column (binary classification) and all other columns are features.
           - Generate the following plots:
             a. Bar plot of the target variable 'Y' distribution using plotly.express.bar() for an interactive version.
             b. Box plots for each numerical feature grouped by 'Y' using matplotlib.pyplot.boxplot() for a static, detailed view.
             c. Correlation heatmap of all numerical features using plotly.express.imshow() for interactivity.
           - Ensure all plots include:
             - Descriptive titles (e.g., 'Distribution of Target Variable Y').
             - Axis labels (e.g., xaxis_title='Feature Name', yaxis_title='Value' for Plotly; plt.xlabel() for Matplotlib).
             - Appropriate color scales or styles (e.g., 'Blues' for heatmap, distinct colors for box plots).

        4. Output the plots as .png files:
           - Save Plotly figures using fig.write_image('filename.png').
           - Save Matplotlib figures using plt.savefig('filename.png') followed by plt.close() to avoid overlap.
           - Name files descriptively (e.g., 'target_distribution.png', 'boxplot_feature1.png', 'correlation_heatmap.png').
           - Ensure the output directory exists or save to the current working directory.

        5. Provide the complete code implementation and analysis results, including:
           - The dataset description from step 1.
           - Descriptive statistics from step 2.
           - Confirmation that all visualizations were generated and saved as .png files with their respective libraries (Plotly or Matplotlib).

        Example code to get started:
        ```python
        import pandas as pd
        import plotly.express as px
        import matplotlib.pyplot as plt

        # Step 1: Load dataset
        df = pd.read_csv('dataset.csv')
        print("First 5 rows:\n", df.head())
        print("Dataset shape:", df.shape)
        print("Column data types:\n", df.dtypes)

        # Step 2: Descriptive statistics
        print("Numerical Statistics:\n", df.describe())
        print("Target Class Distribution:\n", df['Y'].value_counts())

        # Step 3: Visualizations
        # a. Target distribution (Plotly)
        fig1 = px.bar(df['Y'].value_counts(), title='Distribution of Target Variable Y',
                      labels={'index': 'Class', 'value': 'Count'})
        fig1.write_image('target_distribution.png')

        # b. Box plot for a numerical feature (Matplotlib, example: 'feature1')
        plt.figure(figsize=(8, 6))
        for label in df['Y'].unique():
            plt.boxplot(df[df['Y'] == label]['feature1'], positions=[label], widths=0.4,
                        patch_artist=True, label=f'Class {label}')
        plt.title('Box Plot of feature1 by Target Y')
        plt.xlabel('Target (Y)')
        plt.ylabel('feature1 Value')
        plt.legend()
        plt.savefig('boxplot_feature1.png')
        plt.close()

        # c. Correlation heatmap (Plotly)
        corr = df.drop('Y', axis=1).corr()
        fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='Blues',
                         title='Correlation Heatmap of Features')
        fig3.write_image('correlation_heatmap.png')

        # Step 5: Confirmation
        print("Visualizations saved: target_distribution.png (Plotly), boxplot_feature1.png (Matplotlib), correlation_heatmap.png (Plotly)")
        """
        os.environ["GEMINI_API_KEY"]='Use your own'
        model=LiteLLMModel(model_id='gemini/gemini-2.0-flash')
        # Run the visualization agent with the dataframe directly
        visualization_agent = CodeAgent(
            tools=[],
            model=model,
            additional_authorized_imports=[
                "numpy",
                "pandas",
                "matplotlib.pyplot",
                "plotly.express",
                "plotly.graph_objects",
            ],
        )

        # Pass the dataset directly
        result = visualization_agent.run(
            analysis_prompt,
            additional_args={"df": dataset}
        )

        return result
