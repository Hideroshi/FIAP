from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
import sys
import json
import streamlit as st 
import requests

faiss_index_path = "./FAISS"
RAG_AVAILABLE = True

def check_ollama_running():
    try:
        r = requests.get("http://localhost:11434")
        return r.status_code == 200
    except Exception:
        return False

with st.spinner("Verificando conexão com o Ollama..."):
    RAG_AVAILABLE = check_ollama_running()
    if not RAG_AVAILABLE:
        st.error("Ollama não está rodando. O aplicativo funcionará sem RAG.")
    else:
        try:
            with st.spinner("Carregando modelo Ollama e base vetorial..."):
                embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                vectordb = FAISS.load_local(faiss_index_path, embedding, allow_dangerous_deserialization=True)

                llm = Ollama(model="mistral")
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=vectordb.as_retriever(),
                    return_source_documents=True
                )
        except Exception as e:
            st.error(f"Falha ao carregar modelo RAG: {e}")
            st.stop()

def run_stride_rag(componentes):
    if not RAG_AVAILABLE:
        return {"resposta": "RAG não disponível. Verifique se o Ollama está rodando.", "fontes": []}
    
    if not componentes:
        return {"resposta": "", "fontes": []}
    
    prompt = (
        "Considere que os seguintes componentes pertencem a um diagrama de arquitetura de uma aplicação em nuvem:\n"
        f"{', '.join(componentes)}.\n\n"
        "Com base no modelo STRIDE, elabore um parecer técnico que identifique os possíveis riscos de segurança presentes nessa arquitetura.\n\n"
        "Para cada risco identificado:\n"
        "- Explique qual componente está envolvido.\n"
        "- Qual categoria STRIDE se aplica (Spoofing, Tampering, Repudiation, etc).\n"
        "- Descreva o risco de forma clara e objetiva.\n"
        "- Sugira ao menos uma forma de mitigação.\n\n"
        "O texto deve ser conciso, técnico, e estruturado como um relatório profissional de análise de ameaças."
    )

    result = qa_chain.invoke(prompt)
    return {
        "Resposta": result['result'],
        "Fontes": [
            {"pagina": doc.metadata.get('page', '?'), "trecho": doc.page_content[:200]} for doc in result['source_documents']
        ]
    }

