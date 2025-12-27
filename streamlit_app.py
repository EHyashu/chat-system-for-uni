import requests
import streamlit as st

API_URL = "http://localhost:8000/chat"

st.set_page_config(page_title="University AI Assistant", page_icon="🎓")
st.title("🎓 University AI Assistant")
st.write("Ask questions based on your university documents (syllabus, rules, notices, etc.)")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("confidence"):
            confidence_pct = msg["confidence"] * 100
            confidence_color = "green" if confidence_pct >= 70 else "orange" if confidence_pct >= 50 else "red"
            st.caption(f"Confidence: :{confidence_color}[{confidence_pct:.1f}%]")

prompt = st.chat_input("Type your question here...")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        resp = requests.post(API_URL, json={"query": prompt}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "I could not find this information in the university documents.")
        sources = data.get("sources", [])
        confidence = data.get("confidence", 0.0)
        reasoning = data.get("reasoning", "")

        # Build answer with metadata
        answer_with_metadata = answer
        
        if sources:
            answer_with_metadata += "\n\n**📚 Sources:**\n"
            seen_docs = set()
            for s in sources:
                doc = s.get("document_name", "Unknown document")
                if doc not in seen_docs:
                    score = s.get("score", 0.0)
                    rank = s.get("rank", 0) + 1
                    answer_with_metadata += f"- {doc} (Relevance: {score:.2f}, Rank: {rank})\n"
                    seen_docs.add(doc)
        
        if reasoning:
            with st.expander("🔍 System Reasoning"):
                st.caption(reasoning)

    except Exception as e:
        answer_with_metadata = (
            "⚠️ Backend is not available. Please ensure the FastAPI server is running on http://localhost:8000."
        )
        confidence = 0.0

    st.session_state["messages"].append({
        "role": "assistant",
        "content": answer_with_metadata,
        "confidence": confidence,
    })
    with st.chat_message("assistant"):
        st.markdown(answer_with_metadata)
        if confidence > 0:
            confidence_pct = confidence * 100
            confidence_color = "green" if confidence_pct >= 70 else "orange" if confidence_pct >= 50 else "red"
            st.caption(f"Confidence: :{confidence_color}[{confidence_pct:.1f}%]")
