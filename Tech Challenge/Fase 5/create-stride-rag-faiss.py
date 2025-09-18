from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA

import os

# Caminho da pasta com os PDFs
pdf_folder_path = "./STRIDE-PDF"

# Caminho para salvar o índice FAISS
faiss_index_path = "./FAISS"

# Carregar todos os PDFs da pasta
all_pages = []
for filename in os.listdir(pdf_folder_path):
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_folder_path, filename))
        pages = loader.load()
        all_pages.extend(pages)

# Quebrar em chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(all_pages)

# Gerar embeddings
embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectordb = FAISS.from_documents(docs, embedding)
vectordb.save_local(faiss_index_path)

# Conectar ao modelo local via Ollama
llm = Ollama(model="mistral")

# Criar o pipeline RAG
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True
)

# Fazer pergunta
query = "Quais ameaças o STRIDE identifica em sistemas que usam AWS Lambda?"
result = qa_chain.invoke(query)

print("\nResposta:")
print(result['result'])

print("\nFontes:")
for doc in result['source_documents']:
    print(f"- Página: {doc.metadata.get('page', '?')} — {doc.page_content[:200]}...")
