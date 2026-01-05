# RAG para vasta maioria de docs da área de dados - Ingestão e chat.

> Implementação simples, eficiente e **pronta para produção** de um pipeline **RAG (Retrieval‑Augmented Generation)** com ingestão incremental, suporte a múltiplos formatos e interface de debug via Streamlit.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![LangChain](https://img.shields.io/badge/LangChain-v0.2-green)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

---

##  Visão Geral

Este projeto abstrai a complexidade de criar um pipeline de dados para LLMs, oferecendo uma base sólida para **busca semântica, RAG e análise de documentos**.

Ele é dividido em dois módulos principais:

### 🔹 Ingestão (`ingest.py`)
- Varredura automática de diretórios
- Detecção de alterações via **hash MD5** (evita reprocessamento)
- Extração de metadados (ex: ano de referência)
- Geração de embeddings locais
- Persistência em banco vetorial (ChromaDB)

### 🔹 Interface (`app.py`)
- Interface Streamlit simples e objetiva
- Testes de qualidade de retrieval
- Visualização das fontes recuperadas
- Integração com LLMs via Groq (Llama 3)

---

## Funcionalidades

-  **Ingestão incremental** baseada em hash MD5  
-  **Suporte a múltiplos formatos**: PDF, CSV e DOCX  
-  **CSV row‑based inteligente**  
  - Cada linha vira um documento semântico (key‑value)
  - Preserva significado e contexto dos dados tabulares  
-  **Extração automática de metadados** (ex: ano)
-  **Embeddings locais** com Sentence‑Transformers (sem custo de API)
-  **UI de debug** focada em transparência e rastreabilidade

---

##  Como Executar

###  Pré‑requisitos
- Python **3.9+**
- Conta na **Groq** (para inferência com LLM)

---

###  Instalação

1. Clone o repositório:
```bash
git clone https://github.com/rafa-rez/all-docs-RAG.git
cd all-docs-RAG
```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Configure as variáveis de ambiente  
Crie um arquivo `.env` na raiz do projeto:
```env
GROQ_API_KEY=sua_chave_aqui
```

---

##   Uso

###  Passo 1 — Ingestão de Dados

Coloque seus arquivos (`PDF`, `CSV`, `DOCX`) na pasta `dados/` e execute:
```bash
python ingest.py
```

 Isso irá:
- Processar apenas arquivos novos ou modificados
- Gerar embeddings
- Criar o banco vetorial local em `./chroma_db_cache`

---

###  Passo 2 — Rodar a Interface

```bash
streamlit run app.py
```

A interface permitirá:
- Fazer perguntas aos documentos
- Avaliar a qualidade do retrieval
- Ver exatamente **quais trechos foram usados**

---

##  Estrutura do Projeto

```text
.
├── dados/                  # Arquivos de entrada (PDF, CSV, DOCX)
├── chroma_db_cache/        # Banco vetorial persistido (auto-gerado)
├── ingest.py               # Pipeline de ingestão e embeddings
├── app.py                  # Interface Streamlit
├── controle_ingestao.json  # Cache de hashes MD5
├── .env                    # Variáveis de ambiente
└── requirements.txt        # Dependências
```

---

##  Tecnologias

- **Orquestração:** LangChain  
- **Vector Store:** ChromaDB  
- **Embeddings:** Sentence‑Transformers (HuggingFace)  
- **LLM:** Llama 3.1 (via Groq)  
- **Interface:** Streamlit  

---

O foco do projeto é tornar a ingestão de dados não estruturados mais fácil para testes de datasets e lógicas.

---

##  Autor

Desenvolvido por **Rafael Rezende**  
