import os
import time
import streamlit as st
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LC_Pinecone  # Renaming to avoid conflicts
from pinecone import Pinecone, ServerlessSpec

# Initialize connection to Pinecone
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)

# Configuration
cloud = 'aws'
region = 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)
index_name = 'askadocument'

# Check if index already exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        index_name,
        dimension=1536,  # Dimensionality of text-embedding-ada-002
        metric='dotproduct',
        spec=spec
    )

    # Wait for index to be initialized
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

# Load documents
def read_doc(directory):
    file_loader = PyPDFDirectoryLoader(directory)
    documents = file_loader.load()
    return documents

# Chunk data
def chunk_data(data, chunk_size=256):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

# Initialize OpenAI Embeddings
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(model=model_name, openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone Index
index = pc.Index(index_name)
text_field = "text"
vectorstore = LC_Pinecone(index, embed.embed_query, text_field)  # Using the renamed class instance

# User input for query
query = st.text_input("Enter your query:")

# Perform similarity search if query is not empty
if query:
    search_results = vectorstore.similarity_search(query, k=3)
    st.write("Search results:")
    for result in search_results:
        st.write(result)
