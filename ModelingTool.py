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

class ModelingTool(Tool):
    name = "modeling_tool"
    description = "Created Prediction models and the metrics of the said models. Provide the dataframe to analyze."
    inputs = {
        "dataset": {
            "type": "object",
            "description": "Dataframe to analyze and use predictions models over"
        },
        "visualization_type": {
            "type": "string",
            "description": "Type of modeling (classification task) to perform and the models to use if something specific is preferred",
            "nullable": True
        }
    }
    output_type = "object"

    def forward(self, dataset, visualization_type=None):
        # Use the consistent prompt regardless of visualization_type
        analysis_prompt = """
        As an ML Engineer, perform the following tasks using the h2o library on the provided dataset:

        1. Import necessary libraries and initialize h2o:
        ```python
        import h2o
        from h2o.estimators.random_forest import H2ORandomForestEstimator
        from h2o.estimators.gbm import H2OGradientBoostingEstimator
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator
        import pandas as pd

        # Initialize h2o
        h2o.init()

        # Define column types
        col_types = {col: "numeric" for col in dataframe.columns}
        col_types["Y"] = "enum"

        # Convert pandas DataFrame to h2o Frame with specified column types
        df = h2o.H2OFrame(dataframe, column_types=col_types)

        # Identify target and predictor variables
        y = "Y"
        x = df.columns
        x.remove(y)
        x.remove("ID") # Remove ID column

        # Split data into training and test sets
        train, test = df.split_frame(ratios=[0.8], seed=1234)

        # Print train and test columns
        print("Train columns:", train.columns)
        print("Test columns:", test.columns)
        ```

        2. Run AutoML for 10 base models:
        ```python
        from h2o.automl import H2OAutoML
        aml = H2OAutoML(max_models=10, seed=1)
        aml.train(x=x, y=y, training_frame=train)

        # View the AutoML Leaderboard
        lb = aml.leaderboard
        print(lb.head())

        # Get the best model
        best_model = aml.leader

        # Make predictions on the test data
        predictions = best_model.predict(test)

        # Evaluate the model
        performance = best_model.model_performance(test)

        print(performance)

        # Get AUC
        print("AUC:", performance.auc())
        ```

        3. Identify the target variable 'Y' (binary classification: 0 or 1) and predictor variables.

        4. Use AutoML to find the best model:
        - Let's use 10 models and give us the performance metrics for each model.

        5. Train on the training data:
        - This is a classification task since the target column is binary. So make sure to convert it to a factor before training.
        - Make predictions on the test data.

        6. Calculate and print:
        - Other relevant metrics (e.g., AUC, accuracy, precision, recall, F1 macro, F1 score, F1 Micro).

        7. Ensure all operations are performed using h2o functions and methods where possible.

        8. Please provide the complete code implementation and analysis results.

        9. Save the best model using .save in h2o.

        10. Stop the run once the outputs have been displayed and the task has been completed.
        """
        os.environ["GEMINI_API_KEY"]='AIzaSyCxr4mmy9G7ikhes6PDmp2gksYPXA9k1Jo'
        model=LiteLLMModel(model_id='gemini/gemini-2.0-flash')
        # Run the modeling agent with the dataframe directly
        modeling_agent = CodeAgent(
            tools=[],
            model=model,
            additional_authorized_imports=[
                "numpy",
                "pandas",
                "matplotlib.pyplot",
                "statsmodels.api",
                "h2o",
            ],
        )

        # Pass the dataset directly
        result = modeling_agent.run(
            analysis_prompt,
            additional_args={"df": dataset}
        )

        return result