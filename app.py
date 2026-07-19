import streamlit as st
from rag.chatbot import load_rag_chain, ask

st.set_page_config(
    page_title="Edukasi Emas BSI",
    page_icon="🥇",
    layout="centered"
)

st.markdown("""
<style>
    .main { background: #f8f9fa; }
    .chat-header {
        background: linear-gradient(135deg, #00A651, #007A3D);
        color: white; padding: 1.5rem;
        border-radius: 12px; margin-bottom: 1rem;
        text-align: center;
    }
    .source-badge {
        background: #e8f5e9; color: #2e7d32;
        padding: 2px 8px; border-radius: 4px;
        font-size: 0.75rem; margin: 2px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
    <h2>Asisten Emas BSI</h2>
    <p>Edukasi Investasi Emas Digital di Byond by BSI</p>
</div>
""", unsafe_allow_html=True)

if "chain" not in st.session_state:
    with st.spinner("Memuat model AI..."):
        st.session_state.chain = load_rag_chain()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Saya asisten edukasi emas BSI. Saya bisa membantu kamu memahami produk emas di Byond by BSI seperti Tabungan Emas, Cicil Emas, dan Gadai Emas. Ada yang ingin kamu tanyakan?"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if len(st.session_state.messages) == 1:
    st.markdown("**Pertanyaan populer:**")
    cols = st.columns(2)
    questions = [
        "Apa itu Tabungan Emas BSI?",
        "Bagaimana cara beli emas di Byond?",
        "Berapa minimal investasi emas BSI?",
        "Apa keuntungan Cicil Emas BSI?"
    ]
    for i, q in enumerate(questions):
        if cols[i % 2].button(q, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

if prompt := st.chat_input("Tanya tentang emas BSI..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari informasi..."):
            result = ask(st.session_state.chain, prompt)
        st.markdown(result["answer"])

        if result["sources"]:
            st.markdown("**Sumber:**")
            for src in set(result["sources"]):
                st.markdown(f'<span class="source-badge">{src}</span>',
                          unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"]
    })
