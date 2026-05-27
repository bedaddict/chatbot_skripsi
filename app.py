import os
import json
import streamlit as st

# ============================================================
# PAGE CONFIG (HARUS PALING ATAS)
# ============================================================
st.set_page_config(
    page_title="Asisten Emas BSI",
    page_icon="🥇",
    layout="centered"
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHROMA_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw_bsi_data.json")

SYSTEM_PROMPT = """Kamu adalah asisten edukasi investasi emas digital BSI yang ramah, informatif, dan terpercaya.
Kamu membantu pengguna aplikasi Byond by BSI memahami produk emas BSI seperti Tabungan Emas, Cicil Emas, dan Gadai Emas.

ATURAN PENTING:
- Gunakan HANYA informasi dari konteks yang diberikan
- Jika informasi tidak ada dalam konteks, katakan dengan jujur dan sarankan hubungi BSI Call 14040 atau kunjungi bankbsi.co.id
- Jawab dalam Bahasa Indonesia yang ramah dan mudah dipahami
- Gunakan emoji secukupnya agar lebih menarik
- Selalu ingatkan bahwa investasi mengandung risiko
- Jangan memberikan saran keuangan yang bersifat mengikat

Konteks dari database BSI:
{context}"""

# ============================================================
# BUILD INDEX OTOMATIS
# ============================================================
def build_index_if_needed():

    os.makedirs(CHROMA_DIR, exist_ok=True)

    if len(os.listdir(CHROMA_DIR)) > 0:
        return

    st.info("⏳ Membangun database pengetahuan pertama kali...")

    if not os.path.exists(DATA_PATH):
        st.error(f"❌ File data tidak ditemukan: {DATA_PATH}")
        st.stop()

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []

    for item in data:
        if item.get("content") and len(item["content"].strip()) > 50:
            docs.append(
                Document(
                    page_content=item["content"],
                    metadata={
                        "source": item.get("source", ""),
                        "url": item.get("url", ""),
                        "scraped_at": item.get("scraped_at", "")
                    }
                )
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR
    )

build_index_if_needed()

# ============================================================
# STYLING
# ============================================================
st.markdown("""
<style>
    .header-box {
        background: linear-gradient(135deg, #00A651, #007A3D);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .header-box h2 {
        margin: 0;
        font-size: 1.8rem;
    }
    .header-box p {
        margin: 0.3rem 0 0 0;
        opacity: 0.9;
        font-size: 0.95rem;
    }
    .stChatMessage {
        border-radius: 12px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-box">
    <h2>🥇 Asisten Emas BSI</h2>
    <p>Edukasi Investasi Emas Digital di Byond by BSI</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD CHAIN
# ============================================================
@st.cache_resource
def load_chain():

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20}
    )

    groq_key = st.secrets.get(
        "GROQ_API_KEY",
        os.environ.get("GROQ_API_KEY", "")
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=groq_key,
        temperature=0.3,
        max_tokens=1024
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.invoke(x["question"])
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

# ============================================================
# INIT SESSION
# ============================================================
if "chain" not in st.session_state:
    with st.spinner("⏳ Memuat model AI..."):
        st.session_state.chain = load_chain()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """Halo! 👋

Saya asisten edukasi emas BSI.

Saya siap membantu kamu memahami:
- Tabungan Emas
- Cicil Emas
- Gadai Emas

di Byond by BSI 🥇"""
    }]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ============================================================
# QUICK QUESTIONS
# ============================================================
if len(st.session_state.messages) == 1:

    st.markdown("### 💡 Pertanyaan Populer")

    quick_qs = [
        "Apa itu Tabungan Emas BSI?",
        "Bagaimana cara beli emas di Byond?",
        "Berapa minimal investasi emas BSI?",
        "Apa keuntungan Cicil Emas BSI?"
    ]

    cols = st.columns(2)

    for i, q in enumerate(quick_qs):
        if cols[i % 2].button(
            q,
            use_container_width=True,
            key=f"qq_{i}"
        ):
            st.session_state.pending_question = q
            st.rerun()

# ============================================================
# HANDLE QUICK QUESTION
# ============================================================
if "pending_question" in st.session_state:

    q = st.session_state.pop("pending_question")

    st.session_state.messages.append({
        "role": "user",
        "content": q
    })

    with st.spinner("🔍 Mencari jawaban..."):

        answer = st.session_state.chain.invoke({
            "question": q,
            "chat_history": st.session_state.chat_history
        })

    st.session_state.chat_history.append(
        HumanMessage(content=q)
    )

    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    st.rerun()

# ============================================================
# TAMPILKAN CHAT
# ============================================================
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ============================================================
# CHAT INPUT
# ============================================================
if user_input := st.chat_input("Tanya tentang emas BSI..."):

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):

        with st.spinner("🔍 Mencari jawaban..."):

            answer = st.session_state.chain.invoke({
                "question": user_input,
                "chat_history": st.session_state.chat_history
            })

        st.markdown(answer)

    st.session_state.chat_history.append(
        HumanMessage(content=user_input)
    )

    st.session_state.chat_history.append(
        AIMessage(content=answer)
    )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")

st.caption(
    "⚠️ Chatbot ini hanya untuk edukasi. "
    "Investasi mengandung risiko. "
    "BSI Call: 14040"
)
