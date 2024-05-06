
import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os

# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Procurement Manuals and Amendment Compilations in GFR...')
    st.markdown('''
    ## About
    This GPT is designed specifically to assist with understanding procurement rules and identifying deviations in proposals or tender documents related to procurement.
    It uses following documents as a knowledge base to provide detailed insights and comparisons.
    This includes analyzing and summarizing content from manuals and amendments related to procurement guidelines, 
    as well as identifying compliance or discrepancies in tender documents according to the established rules and regulations. 
    This GPT/ChatBot aims to support users in ensuring that procurement activities are carried out in accordance with the legal and regulatory frameworks.


    - [Manual for Procurement of Consultancy & Other Services](https://doe.gov.in/files/manuals_documents/Manual_for_Procurement_of_Consultancy_%26_Other_Services_Updated%20June%2C%202022_1.pdf)
    - [Manual for Procurement of Works](https://www.eicindia.gov.in/WebApp1/resources/PDF/PPM/PPM%2000001.pdf)
    - [Procurement Manual of Goods](https://cdnbbsr.s3waas.gov.in/s316026d60ff9b54410b3435b403afd226/uploads/2023/04/2023042694.pdf)
    - [Compilation of amendments in GFRs, 2017 upto 31.07.2023](https://cdnbbsr.s3waas.gov.in/s316026d60ff9b54410b3435b403afd226/uploads/2023/08/20230830383438708.pdf)

    [Documents Repository](https://drive.google.com/drive/folders/1M4nltzU9T7_pcogUykQWG6NiU4qWjzf8?usp=drive_link)
 
    ''')
    #add_vertical_space(5)
    st.write('Made by LBSNAA for learning purpose](https://www.lbsnaa.gov.in/)')

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('procurement/')
#len(doc)

def chunk_data(docs,chunk_size=800,chunk_overlap=50):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return docs

documents=chunk_data(docs=doc)
#len(documents)

## Embedding Technique Of OPENAI
embeddings=OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])

# vectors=embeddings.embed_query("How are you?")
# len(vectors)

from langchain_pinecone import PineconeVectorStore

vectorstore_from_docs = PineconeVectorStore.from_documents(
    documents,
    index_name='askmydoc2',
    embedding=embeddings
)

## Cosine Similarity Retreive Results from VectorDB
def retrieve_query(query,k=2):
    matching_results=vectorstore_from_docs.similarity_search(query,k=k)
    return matching_results

from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

llm=OpenAI(model_name="gpt-3.5-turbo-instruct",temperature=0.5)
chain=load_qa_chain(llm,chain_type="stuff")

## Search answers from VectorDB
def retrieve_answers(query):
    doc_search=retrieve_query(query)
    print(doc_search)
    response=chain.run(input_documents=doc_search,question=query)
    return response

# our_query = "Please tell me some of the rules mentioned in GFR in bullet points"
# answer= retrieve_answers(our_query)
# print(answer)
st.title("Ask your questions about Procurement manual of Work or Goods or Consultancy or amendments in GFR(2017 to 2023)")

user_question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    answer = retrieve_answers(user_question)
    st.write("Answer:", answer)