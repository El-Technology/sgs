import streamlit as st
from langchain_community.document_loaders import CSVLoader
from langchain_community.llms import OpenAI
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
import os

from dotenv import load_dotenv

# Load environment variables from a .env file for local development
load_dotenv()

openai_api_key = os.getenv("openai_api_key")

st.title('ChatWithCSV')

def load_and_index_data(file_path):
    loader = CSVLoader(file_path=file_path)
    index_creator = VectorstoreIndexCreator()
    return index_creator.from_loaders([loader])

user_question = st.text_input("Enter your question :")
if user_question:
    with st.spinner('Processing...'):
        try:
           
            llm = OpenAI(api_key=openai_api_key)
            docsearch = load_and_index_data('DS_250K_GM_096C_full.csv')
            chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
            response = chain({"question": user_question})
            st.write("Answer:", response['result'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
