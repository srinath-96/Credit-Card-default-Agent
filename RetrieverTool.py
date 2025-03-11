import os
from PyPDF2 import PdfReader
from smolagents import Tool
from langchain_community.retrievers import BM25Retriever
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve relevant parts of research papers related to credit card defaults."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        }
    }
    output_type = "string"

    def __init__(self, pdf_directory, **kwargs):
        super().__init__(**kwargs)
        self.docs = self.process_pdfs(pdf_directory)
        self.retriever = BM25Retriever.from_documents(self.docs, k=5)

    def process_pdfs(self, pdf_directory):
        processed_docs = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
            strip_whitespace=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        for filename in os.listdir(pdf_directory):
            if filename.endswith('.pdf'):
                file_path = os.path.join(pdf_directory, filename)
                content = self.extract_content(file_path)
                chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={"source": filename, "chunk": i}
                    )
                    processed_docs.append(doc)

        return processed_docs

    def extract_content(self, pdf_path):
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return self.clean_text(text)
        except Exception as e:
            print(f"Error extracting content from {pdf_path}: {str(e)}")
            return ""

    def clean_text(self, text):
        # Basic cleaning
        text = text.replace('\x00', '')  # Remove null bytes
        text = ' '.join(text.split())  # Remove extra whitespace
        return text

    def forward(self, query: str) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        docs = self.retriever.invoke(query)
        return "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {doc.metadata['source']} (Chunk {doc.metadata['chunk']}) =====\n" + doc.page_content
                for doc in docs
            ]
        )

# Example usage
# pdf_directory = "/path/to/your/pdf/directory"
# retriever_tool = RetrieverTool(pdf_directory)