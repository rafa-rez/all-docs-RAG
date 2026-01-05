import os
import glob
import json
import hashlib
import re
import logging
import pandas as pd
from dotenv import load_dotenv

# Langchain e Loaders
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, UnstructuredExcelLoader

load_dotenv()

# --- CONFIGURAÇÕES ---
PASTA_RAIZ = "./dados"
PASTA_DB = "./chroma_db_cache"
ARQUIVO_CACHE = "controle_ingestao.json"
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

stats = {"lidos": 0, "ignorados": 0, "erros": 0, "chunks_gerados": 0}

def calcular_hash(caminho):
    """Gera hash MD5 do arquivo para controle de cache."""
    with open(caminho, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def descobrir_ano(texto, nome_arquivo):
    """Tenta extrair ano (2012-2029) do nome do arquivo ou dos primeiros caracteres do texto."""
    match = re.search(r"(201[2-9]|202[0-9])", nome_arquivo)
    if match: return int(match.group(0))
    
    # Procura no início do conteúdo (para PDFs/Docs)
    match_txt = re.search(r"(201[2-9]|202[0-9])", texto[:1000])
    if match_txt: return int(match_txt.group(0))
    
    return 0 # Indefinido

def processar_csv(caminho, nome_arquivo, pasta_pai):
    """Lógica customizada para CSVs de auditoria (row-based)."""
    try:
        try:
            df = pd.read_csv(caminho, sep=";", encoding="utf-8", on_bad_lines="skip", dtype=str)
        except:
            df = pd.read_csv(caminho, sep=";", encoding="latin1", on_bad_lines="skip", dtype=str)
        
        df = df.dropna(how="all").head(10) # LIMITADO PARA TESTE
        df.columns = df.columns.str.strip().str.lower()
        
        docs = []
        ano_fixo = descobrir_ano("", nome_arquivo)

        for _, row in df.iterrows():
            # Tenta achar ano na linha se não achou no arquivo
            ano = ano_fixo
            if ano == 0:
                for col in ["exercicio", "ano", "num_ano_exercicio"]:
                    if col in df.columns and str(row[col]).isdigit():
                        ano = int(row[col])
                        break
            
            conteudo = " | ".join([f"{k.upper()}: {v}" for k, v in row.items() if pd.notnull(v)])
            docs.append(Document(
                page_content=f"FONTE: {pasta_pai}/{nome_arquivo}\nDADOS: {conteudo}",
                metadata={"source": nome_arquivo, "year": ano or 2024, "type": "csv"}
            ))
        return docs
    except Exception as e:
        logger.error(f"Erro no CSV {nome_arquivo}: {e}")
        return []

def carregar_arquivo(caminho):
    """Dispatcher: Seleciona o loader correto baseada na extensão."""
    ext = os.path.splitext(caminho)[1].lower()
    nome = os.path.basename(caminho)
    pasta = os.path.basename(os.path.dirname(caminho))
    
    docs_finais = []
    
    try:
        if ext == ".csv":
            docs_finais = processar_csv(caminho, nome, pasta)
        
        elif ext == ".pdf":
            loader = PyPDFLoader(caminho)
            raw_docs = loader.load()
            # Unifica páginas para extrair ano, idealmente usaria TextSplitter aqui
            for d in raw_docs[:5]: # LIMITADO PARA TESTE (5 págs)
                ano = descobrir_ano(d.page_content, nome)
                d.metadata.update({"source": nome, "year": ano or 2024, "type": "pdf"})
                docs_finais.append(d)

        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(caminho)
            raw_docs = loader.load()
            for d in raw_docs:
                ano = descobrir_ano(d.page_content, nome)
                d.metadata.update({"source": nome, "year": ano or 2024, "type": "docx"})
                docs_finais.append(d)

        elif ext == ".txt":
            loader = TextLoader(caminho, encoding='utf-8')
            raw_docs = loader.load()
            for d in raw_docs:
                ano = descobrir_ano(d.page_content, nome)
                d.metadata.update({"source": nome, "year": ano or 2024, "type": "txt"})
                docs_finais.append(d)
        
        elif ext in [".xlsx", ".xls"]:
            # Excel é complexo, Unstructured ajuda, ou converta para CSV antes
            pass 

    except Exception as e:
        logger.error(f"❌ Falha ao processar {nome}: {e}")
        stats["erros"] += 1
        return []

    return docs_finais

def main():
    logger.info("--- INICIANDO INGESTÃO MULTIMODAL ---")
    
    embeddings = HuggingFaceEmbeddings(model_name=MODELO_EMBEDDING)
    
    # Carrega Cache
    cache = {}
    if os.path.exists(ARQUIVO_CACHE):
        with open(ARQUIVO_CACHE, "r") as f: cache = json.load(f)

    # Busca arquivos recursivamente em qualquer subpasta de dados
    arquivos = []
    for ext in ["*.csv", "*.pdf", "*.docx", "*.txt", "*.xlsx"]:
        arquivos.extend(glob.glob(os.path.join(PASTA_RAIZ, "**", ext), recursive=True))

    buffer = []
    
    for arquivo in arquivos:
        h = calcular_hash(arquivo)
        if cache.get(arquivo) == h:
            stats["ignorados"] += 1
            continue

        docs = carregar_arquivo(arquivo)
        if docs:
            buffer.extend(docs)
            cache[arquivo] = h
            stats["lidos"] += 1
            stats["chunks_gerados"] += len(docs)
            logger.info(f"✅ Lido: {os.path.basename(arquivo)} ({len(docs)} fragmentos)")

        # Batch save
        if len(buffer) >= 500:
            Chroma.from_documents(buffer, embeddings, persist_directory=PASTA_DB)
            buffer = []
            with open(ARQUIVO_CACHE, "w") as f: json.dump(cache, f)

    if buffer:
        Chroma.from_documents(buffer, embeddings, persist_directory=PASTA_DB)
        with open(ARQUIVO_CACHE, "w") as f: json.dump(cache, f)

    logger.info(f"Fim. Lidos: {stats['lidos']} | Chunks: {stats['chunks_gerados']} | Erros: {stats['erros']}")

if __name__ == "__main__":
    main()