# coding: utf-8
# Цей файл є виправленою версією вашого блокноту Asistent.ipynb.
# Ви можете запустити його з терміналу командою: python assistant_fixed.py

# === 1. ІМПОРТ БІБЛІОТЕК ===
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

print("✅ Бібліотеки імпортовано.")

# === 2. ЗАВАНТАЖЕННЯ КЛЮЧІВ API ===
# Завантажуємо змінні з файлу .env.
load_dotenv()

# Зчитуємо GROQ_API_KEY. Цей ключ потрібен для автентифікації в сервісі Groq.
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    print("❌ Помилка: GROQ_API_KEY не знайдено. Будь ласка, додайте його до вашого .env файлу.")
    exit()

# Зчитуємо токен Hugging Face (необов'язково, але корисно для деяких моделей).
hg_token = os.getenv('HF_TOKEN')
if hg_token:
    os.environ['HF_TOKEN'] = hg_token
else:
    print("🔔 Попередження: HF_TOKEN не знайдено у файлі .env.")

print("✅ Ключі API завантажено.")

# === 4. ІНІЦІАЛІЗАЦІЯ МОВНОЇ МОДЕЛІ ===
# Створюємо екземпляр мовної моделі.
# ВИПРАВЛЕНО: Використовуємо 'llama3-8b-8192' — правильну модель для Groq API.
try:
    model_llm = ChatGroq(model='llama-3.1-8b-instant', groq_api_key=groq_api_key)
    print("✅ Мовна модель ChatGroq ініціалізована.")
except Exception as e:
    print(f"❌ Помилка ініціалізації ChatGroq: {e}")
    exit()

# === 5. ІНІЦІАЛІЗАЦІЯ МОДЕЛІ ЕМБЕДИНГІВ ===
# Ембединги — це числові представлення (вектори) тексту.
# Ми використовуємо 'all-MiniLM-L6-v2' — швидку та ефективну модель.
print("⏳ Ініціалізація моделі ембедингів...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print("✅ Модель ембедингів ініціалізована.")

# === 6. СТВОРЕННЯ ДОКУМЕНТІВ ===
# Створюємо список документів, які будуть базою знань для нашого асистента.
documents = [
    Document(
        page_content='Баклан великий (Phalacrocorax carbo) — морський птах, що мешкає на узбережжях Європи та Азії. Довжина тіла до 100 см, розмах крил до 160 см. Живиться рибою, пірнаючи на глибину до 10 метрів.',
        metadata={'origen': 'aves-marinas'}
    ),
    Document(
        page_content='Мартин звичайний (Larus canus) — поширений на морських узбережжях. Гніздиться колоніями на скелях. Всеїдний птах, часто зустрічається біля портів і сміттєзвалищ. Довжина тіла 40-46 см.',
        metadata={'origen': 'aves-marinas'}
    ),
    Document(
        page_content='Олуша північна (Morus bassanus) — великий морський птах з розмахом крил до 180 см. Полює на рибу, пікіруючи з висоти до 40 метрів. Гніздові колонії знаходяться на віддалених острівцях.',
        metadata={'origen': 'aves-marinas'}
    ),
    Document(
        page_content='Качка морська (Melanitta fusca) — пірнаюча качка, що зимує в морських водах. Живиться молюсками та ракоподібними. Самці мають чорне оперення з характерною білою плямою на лобі.',
        metadata={'origen': 'aves-marinas'}
    )
]
print("✅ Документи для бази знань створено.")

# === 8. СТВОРЕННЯ ВЕКТОРНОГО СХОВИЩА ===
# Створюємо векторне сховище (vectorstore) з наших документів в ChromaDB.
print("⏳ Створення векторного сховища ChromaDB...")
vectorstore = Chroma.from_documents(documents, embedding=embeddings)
print("✅ Векторне сховище створено.")

# === 10. СТВОРЕННЯ РЕТРИВЕРА ===
# Ретривер (retriever) "дістає" інформацію з векторного сховища.
# Ми налаштовуємо його повертати 1 (k=1) найбільш релевантний документ.
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':1})
print("✅ Ретривер налаштовано.")

# === 12. СТВОРЕННЯ ШАБЛОНУ ЗАПИТУ (PROMPT) ===
# Готуємо шаблон, який буде відправлено мовній моделі.
# Моделі дається інструкція відповідати, базуючись виключно на наданому контексті.
message_template = """
Відповідаючи на питання використовуй виключно зв'язаний контекст

Питання:
{questions}

Контекст:
{context}
"""

# === 13. СТВОРЕННЯ ОБ'ЄКТА ШАБЛОНУ ===
# ВИПРАВЛЕНО: Використовуємо правильний конструктор `from_template`.
chat_prompt_template = ChatPromptTemplate.from_template(message_template)
print("✅ Шаблон запиту створено.")

# === 14. СТВОРЕННЯ ЛАНЦЮЖКА (CHAIN) ===
# Обєднуємо всі компоненти в єдиний ланцюжок RAG.
chain = (
    {'context': retriever, 'questions': RunnablePassthrough()}
    | chat_prompt_template
    | model_llm
)
print("✅ Ланцюжок RAG зібрано.")

# === 15. ЗАПУСК ЛАНЦЮЖКА ТА ОТРИМАННЯ ВІДПОВІДІ ===
print("\n⏳ Відправка запиту до моделі...")
question = 'яка сама велика тварина на землі?'
response = chain.invoke(question)

print("\n\n==========")
print(f"🧐 Питання: {question}")
print("\n🤖 Відповідь:")
print(response.content)
print("==========")
