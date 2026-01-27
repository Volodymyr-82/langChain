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
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter
from langchain_core.messages import HumanMessage, AIMessage # Додано для ручної роботи з історією

print("✅ Бібліотеки імпортовано.")

# === 2. ЗАВАНТАЖЕННЯ КЛЮЧІВ API ===
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    print("❌ Помилка: GROQ_API_KEY не знайдено. Будь ласка, додайте його до вашого .env файлу.")
    exit()

hg_token = os.getenv('HF_TOKEN')
if hg_token:
    os.environ['HF_TOKEN'] = hg_token
else:
    print("🔔 Попередження: HF_TOKEN не знайдено у файлі .env.")
print("✅ Ключі API завантажено.")

# === 4. ІНІЦІАЛІЗАЦІЯ МОВНОЇ МОДЕЛІ ===
try:
    model_llm = ChatGroq(model='llama-3.1-8b-instant', groq_api_key=groq_api_key)
    print("✅ Мовна модель ChatGroq ініціалізована.")
except Exception as e:
    print(f"❌ Помилка ініціалізації ChatGroq: {e}")
    exit()

# === 5. ІНІЦІАЛІЗАЦІЯ МОДЕЛІ ЕМБЕДИНГІВ ===
print("⏳ Ініціалізація моделі ембедингів...")
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
print("✅ Модель ембедингів ініціалізована.")

# === 6. СТВОРЕННЯ ДОКУМЕНТІВ ===
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
print("⏳ Створення векторного сховища ChromaDB...")
vectorstore = Chroma.from_documents(documents, embedding=embeddings)
print("✅ Векторне сховище створено.")

# === 10. СТВОРЕННЯ РЕТРИВЕРА ===
retriever = vectorstore.as_retriever(search_type='similarity', search_kwargs={'k':1})
print("✅ Ретривер налаштовано.")

# === 11. ІНІЦІАЛІЗАЦІЯ РУЧНОЇ ПАМ'ЯТІ ===
# ВИПРАВЛЕНО: Замість проблемного ConversationBufferMemory, ми використовуємо простий список.
chat_history = []
print("✅ Ручна пам'ять (список) ініціалізована.")

# === 12. СТВОРЕННЯ ШАБЛОНУ ЗАПИТУ (PROMPT) З ІСТОРІЄЮ ===
message_template = ChatPromptTemplate.from_messages(
    [
        ("system", "Відповідаючи на питання використовуй виключно зв'язаний контекст, знайдений в документах. Якщо контекст нерелевантний, посилайся на історію чату. Ти — корисний помічник."),
        MessagesPlaceholder(variable_name="chat_history"), # Сюди буде вставлена історія чату
        ("human", "{question}"),
    ]
)
print("✅ Шаблон запиту з пам'яттю створено.")

# === 13. СТВОРЕННЯ ЛАНЦЮЖКА (CHAIN) З ПАМ'ЯТТЮ ТА RAG ===
rag_chain = (
    RunnablePassthrough.assign(
        context=(lambda x: x["question"]) | retriever
    )
    | message_template
    | model_llm
)
print("✅ Ланцюжок RAG з пам'яттю зібрано.")

# === 14. ІНТЕРАКТИВНИЙ ЗАПУСК ЛАНЦЮЖКА ТА ОТРИМАННЯ ВІДПОВІДІ ===
print("\n========== ІНТЕРАКТИВНИЙ ЧАТ ==========")
print("Введіть 'вихід' або 'exit' для завершення.")

while True:
    user_question = input("\n🧐 Ваше питання: ")
    if user_question.lower() in ["вихід", "exit"]:
        break

    # Створюємо вхідні дані для ланцюжка
    inputs = {"question": user_question, "chat_history": chat_history}
    
    # Відправка питання до ланцюжка та отримання відповіді
    print("⏳ Відправка запиту до моделі...")
    response = rag_chain.invoke(inputs)
    ai_response = response.content
    
    # Оновлюємо нашу ручну історію
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=ai_response))

    print("\n🤖 Відповідь:")
    print(ai_response)
    
print("\n====================================")
print("Діалог завершено.")
print("====================================")
