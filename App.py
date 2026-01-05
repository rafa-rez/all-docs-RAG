import streamlit as st
import time
import pandas as pd
import glob
import re
import os
from datetime import datetime
from dotenv import load_dotenv

# --- LANGCHAIN & CHROMA ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
import chromadb

load_dotenv()

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="RAG Test Environment",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded" # Expandida para mostrar as métricas
)

# --- FUNÇÕES DE BACKEND (MANTIDAS INTACTAS) ---
@st.cache_data(ttl=60)
def carregar_metricas():
    stats = {
        "total_arquivos": 0,
        "tipos": {"PDF": 0, "CSV": 0, "DOC": 0},
        "anos": set(),
        "vetores": 0
    }
    
    if os.path.exists("./dados"):
        arquivos = glob.glob("./dados/**/*.*", recursive=True)
        stats["total_arquivos"] = len(arquivos)
        
        for f in arquivos:
            ext = os.path.splitext(f)[1].lower()
            if ext == ".pdf": stats["tipos"]["PDF"] += 1
            elif ext == ".csv": stats["tipos"]["CSV"] += 1
            elif ext in [".docx", ".doc", ".txt"]: stats["tipos"]["DOC"] += 1
            
            match = re.search(r"(201[2-9]|202[0-9])", os.path.basename(f))
            if match: stats["anos"].add(int(match.group(0)))
    
    try:
        client = chromadb.PersistentClient(path="./chroma_db_cache")
        stats["vetores"] = client.get_collection("langchain").count()
    except:
        stats["vetores"] = 0

    return stats

@st.cache_resource
def setup_rag():
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        vectorstore = Chroma(persist_directory="./chroma_db_cache", embedding_function=embeddings)
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

        template = """
        Use o contexto abaixo para responder a pergunta. 
        Se não souber, diga "Sem dados".
        
        CONTEXTO:
        {context}
        
        PERGUNTA: 
        {question}
        
        RESPOSTA TÉCNICA:
        """
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PromptTemplate.from_template(template)}
        )
        return chain, "Operacional"
    except Exception as e:
        return None, f"Erro: {str(e)}"

# --- ESTADO DA SESSÃO ---
if "logs_execucao" not in st.session_state:
    st.session_state["logs_execucao"] = []

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- INICIALIZAÇÃO ---
metricas = carregar_metricas()
qa_chain, status_msg = setup_rag()

# --- SIDEBAR (MÉTRICAS) ---
with st.sidebar:
    st.title("📊 Monitoramento")
    
    # Status do Sistema com cor visual
    if "Erro" in status_msg:
        st.error(f"Status: {status_msg}")
    else:
        st.success(f"Status: {status_msg}")
    
    st.divider()
    
    st.markdown("### Base de Dados")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Vetores", metricas["vetores"])
    col_s2.metric("Docs Totais", metricas["total_arquivos"])
    
    st.metric("Formatos (PDF / CSV)", f"{metricas['tipos']['PDF']} / {metricas['tipos']['CSV']}")
    st.metric("Anos Detectados", len(metricas["anos"]))
    
    st.divider()
    if st.button("🗑️ Limpar Conversa", type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- ÁREA PRINCIPAL ---
st.title("🤖 RAG Debug Environment")
st.markdown("Ambiente de teste para validação de recuperação de contexto.")

# Abas para separar Chat de Logs Técnicos
tab_chat, tab_logs = st.tabs(["💬 Chat Interativo", "📋 Logs do Sistema"])

# --- ABA 1: CHAT ---
with tab_chat:
    # Container para mensagens (aparece acima do input automaticamente)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # Se for assistente e tiver metadados de fontes salvos (opcional, aqui simplifiquei)
            # você poderia recuperar e mostrar aqui

    # Input do usuário
    if prompt := st.chat_input("Faça uma pergunta aos documentos..."):
        # 1. Mostra msg usuário
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. Processamento
        if qa_chain:
            start_time = datetime.now()
            with st.chat_message("assistant"):
                with st.status("Consultando Vector Store...", expanded=True) as status_box:
                    try:
                        res = qa_chain.invoke({"query": prompt})
                        resposta = res['result']
                        docs = res.get('source_documents', [])
                        fontes_nomes = list(set([os.path.basename(d.metadata.get('source', '')) for d in docs]))
                        
                        status_box.update(label="Processamento concluído!", state="complete", expanded=False)
                        
                        st.markdown(resposta)
                        
                        if fontes_nomes:
                            st.info(f"📚 **Fontes Consultadas:** {', '.join(fontes_nomes)}")
                        
                        # Salva histórico
                        st.session_state.messages.append({"role": "assistant", "content": resposta})

                        # Salva Log
                        st.session_state["logs_execucao"].append({
                            "Horário": start_time.strftime("%H:%M:%S"),
                            "Input": prompt,
                            "Output": resposta,
                            "Fontes": str(fontes_nomes),
                            "Latência": f"{(datetime.now() - start_time).total_seconds():.2f}s"
                        })
                        
                    except Exception as e:
                        status_box.update(label="Erro no processamento", state="error")
                        st.error(str(e))

# --- ABA 2: LOGS ---
with tab_logs:
    st.markdown("### Histórico de Execução (Sessão Atual)")
    
    if st.session_state["logs_execucao"]:
        df_logs = pd.DataFrame(st.session_state["logs_execucao"])
        
        st.dataframe(
            df_logs,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Horário": st.column_config.TextColumn("Horário", width="small"),
                "Input": st.column_config.TextColumn("Pergunta", width="medium"),
                "Output": st.column_config.TextColumn("Resposta", width="large"),
                "Fontes": st.column_config.TextColumn("Sources", width="medium"),
                "Latência": st.column_config.TextColumn("Tempo", width="small"),
            }
        )
        
        if st.button("Limpar Tabela de Logs"):
            st.session_state["logs_execucao"] = []
            st.rerun()
    else:
        st.info("Nenhuma interação registrada ainda.")