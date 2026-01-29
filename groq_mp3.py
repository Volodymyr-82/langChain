import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from gtts import gTTS
import tempfile

load_dotenv()
groq_api_key = os.getenv("GROQ_APP_API_KEY")
llm=ChatGroq(groq_api_key=groq_api_key, model='llama-3.1-8b-instant')
prompt = ChatPromptTemplate.from_template('''
Ти професійний автор пісні. Напишіть пісню в віршах по цій темі: Пиши тільки вірші для виконання. Не добавляй ввічливі відповіді. Тільки текст пісні.
Тема: {input}

Пісня повинна мати строфи, бути оригінальною і ліричною. Нехай римуйтеся, а закінчення рядка с легка витягнуті. Не користуйся **.
''')
st.set_page_config(page_title="🎤 Генератор пісень з озвучкой", layout="centered")
st.title("🎶 AI Творець пісень + Голос")

theme = st.text_input("🎵 Введіть тему или фразу для пісні (наприклад, зима, заборонене кохання и т.д.)")

if theme and st.button("🎼 Створити AI пісню"):
    with st.spinner("🧠 Llama3 пише вірш..."):
        chain=prompt | llm
        lyrics=chain.invoke({'input': theme})
        st.markdown("### 🎤 Сгенерированный текст:")
        st.text_area("Текст песни:", lyrics.content, height=200)
        with st.spinner("🎧 Синтезируем песню..."):
            tts = gTTS(text=lyrics.content, lang='uk')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
                tts.save(tmp_file.name)
                audio_path = tmp_file.name
        st.success("✅ Пісня готова!")
        st.audio(audio_path, format="audio/mp3")

        with open(audio_path, "rb") as f:
            st.download_button(
                label="💾 Скачати MP3",
                data=f,
                file_name="generated_song.mp3",
                mime="audio/mp3"
            )