import openai
import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
import streamlit as st


# Sidebar contents
with st.sidebar:
    st.title('💬 LLM Chat App on Annual reports of Meity for different financial years')
    st.markdown('''
    ## About
    This chatbot helps in answering questions related to Annual reports from year 2017-23

    [Documents Repository](https://drive.google.com/drive/folders/12CviwBib5xdWy3pW5trrOJxPbZFht2cn?usp=drive_link)
 
    ''')
    #add_vertical_space(5)
    st.write('Made by LBSNAA for learning purpose](https://www.lbsnaa.gov.in/)')

def read_doc(directory):
    file_loader=PyPDFDirectoryLoader(directory)
    documents=file_loader.load()
    return documents

doc=read_doc('Ministry of Electronics and Information Technology (MEITY)/')
len(doc)

def chunk_data(data, chunk_size=256):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
    chunks = text_splitter.split_documents(data)
    return chunks

def print_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    print(f'Total Tokens: {total_tokens}')
    print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')


pc = pinecone.Pinecone()

def insert_or_fetch_embeddings(index_name, chunks):
    # importing the necessary libraries and initializing the Pinecone client
    import pinecone
    from langchain_community.vectorstores import Pinecone
    from langchain_openai import OpenAIEmbeddings
    from pinecone import PodSpec

    from pinecone import ServerlessSpec

      
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well

    # loading from existing index
    if index_name in pc.list_indexes().names():
        print(f'Index {index_name} already exists. Loading embeddings ... ', end='')
        vector_store = Pinecone.from_existing_index(index_name, embeddings)
        print('Ok')
    else:
        # creating the index and embedding the chunks into the index 
        print(f'Creating index {index_name} and embeddings ...', end='')

        # creating a new index
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )

        # processing the input documents, generating embeddings using the provided `OpenAIEmbeddings` instance,
        # inserting the embeddings into the index and returning a new Pinecone vector store object. 
        vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
        print('Ok')
        
    return vector_store


def ask_and_get_answer(vector_store, q, k=3):
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    answer = chain.invoke(q)
    return answer


# import warnings
# warnings.filterwarnings('ignore')

data = read_doc('Ministry of Electronics and Information Technology (MEITY)/')
# print(data[1].page_content)
# print(data[10].metadata)

print(f'You have {len(data)} pages in your data')
print(f'There are {len(data[20].page_content)} characters in the page')

chunks = chunk_data(data)
print(len(chunks))
# print(chunks[10].page_content)

index_name = 'askadocument'
vector_store = insert_or_fetch_embeddings(index_name=index_name, chunks=chunks)


st.title("Ask your questions about Procurement manual of Work or Goods or Consultancy or amendments in GFR(2017 to 2023)")

user_question = st.text_input("Ask your question:")

if st.button("Get Answer"):
    answer = ask_and_get_answer(user_question)
    st.write("Answer:", answer)