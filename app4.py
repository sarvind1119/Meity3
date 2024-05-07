from pinecone import ServerlessSpec
import os
from pinecone import Pinecone
import streamlit as st
# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.environ.get('PINECONE_API_KEY')
pc = Pinecone(api_key=api_key)
from pinecone import ServerlessSpec

cloud = 'aws'
region = 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = 'arvind1'
import time
# switch back to normal index for langchain
index = pc.Index(index_name)
from langchain.embeddings.openai import OpenAIEmbeddings
index = pc.Index(index_name)
# wait a moment for connection
time.sleep(1)

index.describe_index_stats()


#Creating a Vector Store and Querying
from langchain.embeddings.openai import OpenAIEmbeddings

# get openai api key from platform.openai.com
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

#initialize the vector store
from langchain.vectorstores import Pinecone

text_field = "text"

# switch back to normal index for langchain
index = pc.Index(index_name)

vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

#querying using vectorstore.similarity_search
#query = "who was Arvind kejriwal?"

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

from langchain.chains import RetrievalQAWithSourcesChain

qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

def retrieval_answer(query):
    qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type='stuff',
    retriever=vectorstore.as_retriever(),
    )
    query = query
    result = qa.run(query)
    return result
# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Meity Annual reports...')
    st.markdown('''
    ## About
    This GPT helps in answering questions related to annual reports of MEITY from year 2017 to 2023



    [Documents Repository](https://drive.google.com/drive/folders/12CviwBib5xdWy3pW5trrOJxPbZFht2cn?usp=sharing)
 
    ''')
    #add_vertical_space(5)
    st.write('Made by LBSNAA for learning purpose](https://www.lbsnaa.gov.in/)')


st.title("Ask your questions about Meity Annual reports")

def main():
    #st.title("Question and Answering App powered by LLM and Pinecone")

    text_input = st.text_input("Ask your query...") 
    if st.button("Ask Query"):
        if len(text_input)>0:
            st.info("Your Query: " + text_input)
            answer = retrieval_answer(text_input)
            st.success(answer)

if __name__ == "__main__":
    main()
#response = qa_with_sources(query)

#streamlit -------------------------------------------------------------------

# # Sidebar contents
# with st.sidebar:
#     st.title('💬 LLM Chat App on Procurement Manuals and Amendment Compilations in GFR...')
#     st.markdown('''
#     ## About
#     This GPT helps in answering questions related to annual reports of MEITY from year 2017 to 2023



#     [Documents Repository](https://drive.google.com/drive/folders/12CviwBib5xdWy3pW5trrOJxPbZFht2cn?usp=sharing)
 
#     ''')
#     #add_vertical_space(5)
#     st.write('Made by LBSNAA for learning purpose](https://www.lbsnaa.gov.in/)')


# st.title("Ask your questions about Meity Annual reports")

# user_question = st.text_input("Ask your question:")

# if st.button("Get Answer"):
#     answer = qa(user_question)
#     #answer = ask_and_get_answer(vector_store,user_question,k=3)
#     st.write("Answer:", answer)
