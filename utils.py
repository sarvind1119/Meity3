from sentence_transformers import SentenceTransformer
import pinecone
import openai
import os
import streamlit as st
from langchain.vectorstores import Pinecone
openai.api_key = os.environ.get('OPENAI_API_KEY') ## find at platform.openai.com
from sentence_transformers import SentenceTransformer
import torch
from pinecone import Pinecone, ServerlessSpec
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
model


pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
index = pc.Index('askadocument')

def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(input_em,
    top_k=3,
    include_values=True)
    return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']

def query_refiner(conversation, query):

    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    temperature=0.7,
    max_tokens=256,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )
    return response['choices'][0]['text']

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string