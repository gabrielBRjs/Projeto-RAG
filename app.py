import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# CONFIGURAÇÃO DA CHAVE DA API (Apenas para o Chatbot responder)
os.environ["GOOGLE_API_KEY"] = "AQ.Ab8RN6KX9PcddmveFJbd6G9Qk6GZiXMCjm258X0tIT0N83rJ0Q"

st.set_page_config(page_title="Chatbot Contestado - RAG", page_icon="📜")
st.title("📜 Chatbot Inteligente - Guerra do Contestado")
st.subheader("Base de Conhecimento RAG baseada na obra de Sebastião Luiz Alves")

# Função auxiliar para formatar os documentos recuperados
def formatar_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 1. CARREGAMENTO E INDEXAÇÃO DO PDF (RAG - Etapa de Indexação)
@st.cache_resource
def inicializar_sistema_rag():
    pdf_path = "contestado.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"Arquivo '{pdf_path}' não encontrado na raiz do projeto!")
        return None
    
    # Carrega o documento PDF
    loader = PyPDFLoader(pdf_path)
    paginas = loader.load()
    
    # Divide o texto em pedaços pequenos (Chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(paginas)
    
    # Usando SentenceTransformers local - Livre de erros de API 404!
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    banco_vetores = FAISS.from_documents(chunks, embeddings)
    
    # Retorna o buscador (Retriever) trazendo os 3 trechos mais relevantes
    return banco_vetores.as_retriever(search_kwargs={"k": 3})

retriever = inicializar_sistema_rag()

if retriever:
    # 2. CONFIGURAÇÃO DA IA E PROMPT ANTI-ALUCINAÇÃO (RAG - Recuperação e Geração)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    system_prompt = (
        "Você é um historiador virtual especialista na Guerra do Contestado.\n"
        "Sua tarefa é responder à pergunta do usuário baseando-se estritamente nos trechos do livro fornecidos abaixo.\n"
        "Resuma as informações de forma clara e direta.\n"
        "Se o contexto abaixo não trouxer nenhuma informação sobre o que foi perguntado, responda exatamente: "
        "'Não sei. Esta informação não foi encontrada no documento fornecido.'\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])
    
    # Construção da esteira RAG usando LCEL
    rag_chain = (
        {"context": retriever | formatar_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # INTERFACE DO USUÁRIO (Streamlit)
    pergunta = st.text_input("Faça uma pergunta sobre a Guerra do Contestado:")
    
    if pergunta:
        with st.spinner("Pesquisando no livro do Contestado..."):
            # Executa a busca e gera a resposta
            resposta_texto = rag_chain.invoke(pergunta)
            
            st.markdown("### 🤖 Resposta:")
            st.write(resposta_texto)
            
            # Recupera os documentos separadamente para mostrar as fontes ao professor
            documentos_fontes = retriever.invoke(pergunta)
            with st.expander("📄 Trechos consultados no PDF (Fontes de Verdade):"):
                for doc in documentos_fontes:
                    num_pagina = doc.metadata.get('page', 0) + 1
                    st.write(f"**Página {num_pagina}:** {doc.page_content}")