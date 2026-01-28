import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain

from dotenv import load_dotenv
import tempfile

load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

llm=ChatGroq(api_key=groq_api_key, model='llama-3.1-8b-instant')
prompt_template = ChatPromptTemplate.from_template(
    """
    Надай розлогу відповідь тільки на основі наданого контексту.
    Якщо відповіді немає в контексті, напиши, що інформація відсутня.

    <context>
    {context}
    </context>

    Запит: {input} 
    """
)
st.set_page_config(page_title='RAG', layout='centered')
st.title('')

pdf_file=st.file_uploader(' ', type='pdf')
if 'vector_store' not in st.session_state:
    st.session_state['vector_story']=None
if pdf_file and st.button('uload pdf'):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        st.write(pdf_file.read())
        tmp_path=tmp.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs=loader.load()

        splitter=RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
        chunks=splitter.split_documents(docs)

        embeddings=SentenceTransformerEmbeddings(model='all-MiniLM-L6-v2')

        vector_store=FAISS.from_documents(chunks, embeddings=embeddings)
        st.session_state.vector_store=vector_store
        st.success(f'PDF: {len(chunks)} documents')
    except Exception as e:
        st.error(f'Error: {e}')

user_input=st.text_area('')
if user_input and st.session_state.vector_store:
    try:
        retrieval=st.session_state.vector_store.as_retriever()
        doc_chain=create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain=create_retrieval_chain(retrieval, doc_chain)

        with st.spinner(''):
            result=retrieval_chain.invoke({'input': user_input})
            st.markdown('### respons')
            st.write(['answer'])
    except Exception as e:
        st.error(f'Error: {e}')



