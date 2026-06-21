import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# CONFIGURAÇÃO DA CHAVE DA API
# Substitua pelo seu token gerado no Google AI Studio
os.environ["GOOGLE_API_KEY"] = "COLE_SUA_CHAVE_DO_GEMINI_AQUI"

st.set_page_config(page_title="Chatbot Contestado - RAG", page_icon="📜")
st.title("📜 Chatbot Inteligente - Guerra do Contestado")
st.subheader("Base de Conhecimento RAG baseada na obra de Sebastião Luiz Alves")

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
    
    # Divide o texto em pedaços pequenos (Chunks) para facilitar a busca
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(paginas)
    
    # Transforma os pedaços de texto em vetores (Embeddings) e salva no banco FAISS
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    banco_vetores = FAISS.from_documents(chunks, embeddings)
    
    # Retorna o buscador (Retriever) configurado para trazer os 3 trechos mais relevantes
    return banco_vetores.as_retriever(search_kwargs={"k": 3})

retriever = inicializar_sistema_rag()

if retriever:
    # 2. CONFIGURAÇÃO DA IA E PROMPT ANTI-ALUCINAÇÃO (RAG - Recuperação e Geração)
    # Temperatura zero garante que o modelo seja factual e não invente coisas
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Prompt rígido para cumprir a especificação de não alucinar fora do texto
    system_prompt = (
        "Você é um historiador virtual especialista na Guerra do Contestado.\n"
        "Sua tarefa é responder à pergunta do usuário utilizando APENAS os trechos do livro fornecidos abaixo.\n"
        "Se a informação não estiver descrita explicitamente no contexto abaixo, responda exatamente: "
        "'Não sei. Esta informação não foi encontrada no documento fornecido.'\n"
        "Nunca tente inventar fatos históricos ou complementar com conhecimento externo.\n\n"
        "Contexto:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Cria as esteiras (Chains) de execução do RAG
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # INTERFACE DO USUÁRIO (Streamlit)
    pergunta = st.text_input("Faça uma pergunta sobre a Guerra do Contestado:")
    
    if pergunta:
        with st.spinner("Pesquisando no livro do Contestado..."):
            # Executa a busca e gera a resposta
            resposta = rag_chain.invoke({"input": pergunta})
            
            st.markdown("### 🤖 Resposta:")
            st.write(resposta["answer"])
            
            # Mostra as fontes para provar ao professor que o RAG funcionou de verdade
            with st.expander("📄 Trechos consultados no PDF (Fontes de Verdade):"):
                for doc in resposta["context"]:
                    # Soma 1 porque o índice da página no código começa em 0
                    num_pagina = doc.metadata.get('page', 0) + 1
                    st.write(f"**Página {num_pagina}:** {doc.page_content}")