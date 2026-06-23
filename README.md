# 📜 Chatbot Inteligente - Guerra do Contestado (RAG System)

Este é um sistema de **RAG (Retrieval-Augmented Generation)** desenvolvido como projeto acadêmico. O objetivo é criar um Chatbot especialista na **Guerra do Contestado**, utilizando como base de conhecimento estrita a obra *"Guerra do Contestado: Linha do Tempo"* do historiador Sebastião Luiz Alves, obtida no acervo digital da Biblioteca Pública de Santa Catarina.

O sistema conta com uma trava anti-alucinação para garantir que a inteligência artificial responda apenas com informações contidas no documento fornecido.

---

## 🚀 Funcionalidades

*   **Leitura e Fragmentação de PDF:** Processamento automático do livro histórico e divisão em blocos de texto (*chunks*).
*   **Embeddings Locais:** Geração de vetores semânticos localmente na máquina utilizando o modelo `all-MiniLM-L6-v2` da HuggingFace (via SentenceTransformers).
*   **Banco de Vetores FAISS:** Armazenamento e busca eficiente por proximidade de contexto, permitindo que o sistema entenda o significado das perguntas.
*   **Prompt Anti-Alucinação:** Configuração rígida que impede o modelo de inventar fatos ou utilizar conhecimento externo ao PDF.
*   **Rastreabilidade de Fontes:** Interface que exibe a resposta da IA e, de forma transparente, as páginas e os trechos exatos do livro que foram consultados.
*   **Interface Web:** Interface gráfica amigável desenvolvida em Streamlit.

---

## 🛠️ Tecnologias Utilizadas

*   **Python 3.14**
*   **Streamlit** (Interface de usuário)
*   **LangChain / LCEL** (Orquestração do fluxo RAG)
*   **Google Gemini API** (Modelo `gemini-2.5-flash` para geração de respostas)
*   **FAISS** (Banco de dados vetorial)
*   **HuggingFace / SentenceTransformers** (Embeddings locais)

---
