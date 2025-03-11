from dotenv import load_dotenv
from typing import Any, Dict
from smolagents import CodeAgent, HfApiModel, Tool, ToolCallingAgent,LiteLLMModel,DuckDuckGoSearchTool
from smolagents.default_tools import VisitWebpageTool
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
from DataVisualizationTool import DataVisualizationTool
from ModelingTool import ModelingTool
from RetrieverTool import RetrieverTool
import sys
# CSS for my cool interface
CSS = """
<style>
/* General page styling */
body {
    font-family: 'Google Sans', 'Roboto', sans-serif;
    background-color: #f9f9f9;
}

/* Header styling */
header {
    background-color: #fff;
    padding: 10px 20px;
    border-bottom: 1px solid #e0e0e0;
    position: fixed;
    top: 0;
    width: 100%;
    z-index: 1000;
}

/* Chat container */
.chat-container {
    margin-top: 60px;
    max-width: 800px;
    margin-left: auto;
    margin-right: auto;
    padding: 20px;
}

/* Chat message bubbles */
.stChatMessage {
    border-radius: 12px;
    padding: 10px 15px;
    margin: 5px 0;
    max-width: 70%;
}

/* User message (right-aligned, blue background) */
.stChatMessage.user {
    background-color: #4285f4;
    color: white;
    margin-left: auto;
}

/* Assistant message (left-aligned, light gray background) */
.stChatMessage.assistant {
    background-color: #f1f3f4;
    color: black;
}

/* Input box styling */
.stChatInput {
    position: fixed;
    bottom: 20px;
    width: 800px;
    margin-left: auto;
    margin-right: auto;
    left: 0;
    right: 0;
    background-color: #fff;
    border: 1px solid #dadce0;
    border-radius: 24px;
    padding: 10px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
}
</style>
"""

def get_conversation_context(history, num_messages=10):
    return "\n".join(history[-num_messages:])

def credit_risk_assistant(dataframe):
    # Inject custom CSS
    st.markdown(CSS, unsafe_allow_html=True)

    # Header
    st.markdown('<header><h2 style="margin: 0; font-size: 24px;">Credit Risk Assistant</h2></header>', unsafe_allow_html=True)

    # Initialize tools
    visualization_tool = DataVisualizationTool()
    modeling_tool = ModelingTool()  
    retriever_tool = RetrieverTool(pdf_directory="/Users/srinathmurali/Desktop/untitled folder 2")
    search = DuckDuckGoSearchTool()
    os.environ["GEMINI_API_KEY"]='I was too lazy to use a .env'
    model=LiteLLMModel(model_id='gemini/gemini-2.0-flash')
    primary_agent = CodeAgent(
        tools=[visualization_tool, modeling_tool, retriever_tool, search],
        model=model,  
        additional_authorized_imports=["pandas"],
        add_base_tools=True
    )

    # Initialize session state
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = {
            "history": [],
            "current_df": dataframe,
            "last_visualization": None,
            "last_modeling": None,
            "is_greeting_done": False
        }

    # Welcome message 
    if not st.session_state.conversation_state["is_greeting_done"]:
        with st.chat_message("assistant", avatar="🤖"):
            st.write("I’m here to help with credit risk analysis. What’s on your mind?")
        st.session_state.conversation_state["history"].append("Bot: I’m here to help with credit risk analysis. What’s on your mind?")
        st.session_state.conversation_state["is_greeting_done"] = True

    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.conversation_state["history"]:
        if message.startswith("User: "):
            with st.chat_message("user", avatar="👤"):
                st.write(message.replace("User: ", ""))
        elif message.startswith("Bot: "):
            with st.chat_message("assistant", avatar="🤖"):
                st.write(message.replace("Bot: ", ""))
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input at the bottom
    user_input = st.chat_input("Type your message here...")
    if user_input:
        if user_input.lower() in ["exit", "quit", "bye"]:
            with st.chat_message("assistant", avatar="🤖"):
                st.write("Goodbye!")
                st.write("Shutting down the app...")  # Optional feedback
                sys.exit(0)
            st.session_state.conversation_state["history"].append("Bot: Goodbye!")
            return

    
        st.session_state.conversation_state["history"].append(f"User: {user_input}")
        context = get_conversation_context(st.session_state.conversation_state["history"])

        prompt = f"""
        Conversation history:
        {context}

        User's latest message: "{user_input}"

        You are a Credit Risk Intelligence Platform assistant. Your goal is to assist with credit risk analysis seamlessly, transitioning between tools based on the user's query.

        Available tools and data:
        - 'dataset': A pandas dataframe with credit risk data (columns like credit score, income, payment history, etc.).
        - 'data_visualizer': Generates visualizations from the dataset (returns a description or file path). Use it once per query and stop after presenting the result.
        - 'Modeling': Runs ML models on the dataset, capped at 3 steps (returns best model details). Use it once per query and stop after presenting the result.
        - 'retriever': Searches research papers on credit card defaults (returns relevant text chunks). Use it once per query and stop after presenting the result.
        - 'DuckDuckGoSearch': Searches the web for general info (returns search results). Use it once per query and stop after presenting the result.

        Current state:
        - Previous visualizations: {st.session_state.conversation_state["last_visualization"] if st.session_state.conversation_state["last_visualization"] else "None"}
        - Previous modeling: {st.session_state.conversation_state["last_modeling"] if st.session_state.conversation_state["last_modeling"] else "None"}

        Instructions:
        - Interpret the user’s query naturally and decide which tool (if any) to use based on context and intent.
        - If the query is about the dataset (e.g., "What’s in it?" or "What factors matter?"), inspect 'dataset' and respond with insights.
        - If the query suggests visualization (e.g., "Show me trends" or "Visualize defaults"), use 'data_visualizer' ONCE, present the result, and STOP—do not re-run unless explicitly asked again.
        - If the query suggests modeling (e.g., "Predict defaults" or "Run a model"), use 'Modeling' ONCE, present the result (noting it’s from 3 steps), and STOP—do not re-run unless explicitly asked again.
        - If the query is research-oriented (e.g., "What causes defaults?"), use 'retriever' or 'DuckDuckGoSearch' ONCE based on specificity, present the result, and STOP—do not re-run unless explicitly asked again.
        - Be conversational, integrate the tool result into your response, and always ask what the user wants to do next.
        - After presenting a tool’s result, do not invoke any tool again until the user provides a new query with a clear intent.
        - Avoid unnecessary confirmations unless the query is truly unclear.
        - Do not repeat previous responses verbatim—build on the conversation.

        Respond directly to the user's latest message in a natural, chat-like tone.
        """

        # Run the agent and get the response
        response = primary_agent.run(prompt, additional_args={"dataset": st.session_state.conversation_state["current_df"]})

        # Update state with results (only if new)
        if "Visualization generated" in response and not st.session_state.conversation_state["last_visualization"]:
            st.session_state.conversation_state["last_visualization"] = visualization_tool.forward(st.session_state.conversation_state["current_df"])
        elif "Modeling completed" in response and not st.session_state.conversation_state["last_modeling"]:
            st.session_state.conversation_state["last_modeling"] = modeling_tool.forward(st.session_state.conversation_state["current_df"])

        # Add bot response to history and display it
        st.session_state.conversation_state["history"].append(f"Bot: {response}")
        with st.chat_message("assistant", avatar="🤖"):
            st.write(response)

if __name__ == "__main__":
    # Example usage: Load your dataframe here
    import pandas as pd
    from ucimlrepo import fetch_ucirepo

# fetch dataset
    dataset = fetch_ucirepo(id=350)
    d=dataset['data']['original']# Update with your actual data path
    credit_risk_assistant(d)
