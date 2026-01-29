# Цей скрипт реалізує систему "Retrieval-Augmented Generation" (RAG) для обробки PDF-файлів.
# Він дозволяє користувачам завантажувати PDF-документи, розбивати їх на фрагменти,
# створювати векторне сховище для цих фрагментів, а потім задавати питання.
# Відповіді генеруються великою мовною моделлю (LLM) з використанням контексту,
# отриманого з завантажених PDF-документів.

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
# Завантаження API ключа Groq з змінних оточення
groq_api_key = os.getenv('GROQ_API_KEY')

# Ініціалізація великої мовної моделі (LLM) від Groq
llm = ChatGroq(api_key=groq_api_key, model='llama-3.1-8b-instant')

# Шаблон промпта для LLM, який включає контекст та запит користувача
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
st.set_page_config(page_title='RAG система для PDF', layout='centered')
st.title('📚 RAG система для PDF') # Додаємо емодзі до заголовка

# Віджет для завантаження PDF файлу з емодзі
pdf_file = st.file_uploader('⬆️ Завантажте PDF файл', type='pdf')

if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None

if pdf_file is not None:
    # Кнопка для обробки завантаженого PDF-файлу
    if st.button('✨ Обробити PDF'):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name

        try:
            # Завантаження PDF за допомогою PyPDFLoader
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()

            # Розбиття документів на фрагменти (chunks)
            splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
            chunks = splitter.split_documents(docs)

            # Створення вбудовувань (embeddings) для фрагментів тексту
            embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')

            # Створення векторного сховища (FAISS) з фрагментів та їх вбудовувань
            vector_store = FAISS.from_documents(chunks, embeddings)
            st.session_state.vector_store = vector_store
            st.success(f'✅ PDF оброблено: {len(chunks)} фрагментів')

            # Видаляємо тимчасовий файл
            os.unlink(tmp_path)

        except Exception as e:
            st.error(f'❌ Помилка: {e}')

# Віджет для введення запиту користувачем
user_input = st.text_input('Задайте питання:')

if user_input and st.session_state.vector_store:
    try:
        # Ініціалізація ретрівера для пошуку відповідних документів у векторному сховищі
        retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Створення ланцюжка для комбінування документів
        doc_chain = create_stuff_documents_chain(llm, prompt_template)
        
        # Створення ланцюжка для RAG, який поєднує ретрівер і ланцюжок документів
        retrieval_chain = create_retrieval_chain(retriever, doc_chain)

        # Відображення спінера під час пошуку відповіді
        with st.spinner('💬 Шукаю відповідь...'):
            # Виклик ланцюжка RAG з запитом користувача
            result = retrieval_chain.invoke({"input": user_input})
            st.markdown('### 💡 Відповідь:') # Додаємо емодзі до заголовка відповіді
            st.write(result['answer'])

    except Exception as e:
        st.error(f'❌ Помилка: {e}')