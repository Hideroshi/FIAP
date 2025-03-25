import json
from langchain_community.document_loaders import JSONLoader
from pathlib import Path
from pprint import pprint
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch
from langchain.schema import Document
from langchain_chroma import Chroma
from uuid import uuid4

def retrieve_processed_file_data(processed_file = "trn_processed.json"):
    """
    Retrieves ``processed_file`` (a ``json`` file that contains a list 
    of JSON documents).
    """
    print("\nCarregando dados do arquivo processado...")
    processed_file_data = json.loads(Path(processed_file).read_text())
    print(f"Total de registros gerados:{len(processed_file_data)}")
    
    print("Exemplo de registro: ")
    pprint(processed_file_data[0])

    return processed_file_data

def break_json_data_In_chunks(json_data):
    """
    Breaks ``json_data`` (a ``json`` data Python object) into text chunks.
    """
    print("\nQuebrando documentos JSON em chunks...")
    # Configura o splitter
    splitter = CharacterTextSplitter(
        separator       = "\n",
        chunk_size      = 300,
        chunk_overlap   = 30,
        length_function = len,
    )
    # Processa cada documento JSON individualmente
    chunks = []
    for item in tqdm(json_data, desc="Dividindo documentos JSON"):
        # Extrai o texto do item (produto + descrição)
        text = f"Product: {item.get('product')} - Description: {item.get('description')}"            
        # Divide cada item separadamente
        item_chunks = splitter.split_text(text)
        chunks.extend(item_chunks)

    print(f"\nTotal de chunks gerados: {len(chunks)}")

    # Mostra exemplo de chunk
    print("\nExemplo de chunk:")
    print(chunks[0])

    return chunks

def create_embeddings_model():
    """
    Generates ``embeddings_model`` (a model of embeddings).
    """
    print("\nCriando modelo de embeddings...")
    # Configura e gera os embeddings
    model_name    = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs  = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}

    # Cria o modelo de embeddings
    print("\nCriando modelo de embeddings...")
    embeddings_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings_model

def generate_embeddings_for_chunks(json_chunks, embeddings_model):
    """
    Generates ``embeddings`` for ``json_chunks`` (a list of ``json`` 
    stringyfied string) and a given ``embeddings_model`` (a model of embeddings).
    """
    print("\nGerando embeddings para os chunks...")

    # Gera embeddings para os chunks
    print("\nGerando embeddings...")
    embeddings = embeddings_model.embed_documents(json_chunks)
    print(f"\nTotal de embeddings gerados: {len(embeddings)}")

    # Mostra exemplo dos primeiros embeddings
    print("\nExemplo de embedding:")
    pprint(embeddings[0])

    return embeddings


def convert_json_data_to_lc_documents(json_data):
    """
    Converts ``json_data`` (a list of ``json`` data Python objects) into  
    ``lc_documents`` (a list of LangChain ``json`` documents).
    """
    print("\nConvertendo json_data em lc_documents...\n")
    # Converte os dicionários em objetos Document
    lc_documents = []
    for item in json_data:
        lc_document = Document(
            page_content = f"Product: {item['product']} - Description: {item['description']}",
            metadata     = {"product": item['product']}
        )
        lc_documents.append(lc_document)
    print(f"\nTotal de lc_documents  gerados: {len(lc_documents)}")

    # Mostra exemplo dos primeiros embeddings
    print("\nExemplo de lc_document:")
    pprint(lc_documents[0])    
    return lc_documents    

def create_vector_store(embeddings_model):
    """
    Creates a ``vector_store`` (a vector database) using an ``embeddings_model`` 
    (a model of embeddings).
    """
    print("\nCriando vector_store para o embeddings_model...")
    # Cria o vector store com o modelo de embeddings
    vector_store = Chroma(
        collection_name    = "product_description",
        embedding_function = embeddings_model,
        persist_directory  = "./chroma_langchain_db",  
    )
    print("\nvector_store criada!")
    return vector_store


def populate_vector_store(vector_store, lc_documents, embeddings):
    """
    Populates a ``vector_store`` (a vector database) for ``lc_documents`` 
    (a list of LangChain ``json`` documents).
    """
    print("\nPopulando vector_store...")

    # Gera UUIDs para cada documento
    print("\nGerando UUIDs para cada documento...")
    uuids = [str(uuid4()) for _ in range(len(lc_documents))]    

    # Adiciona os documentos
    print("\nAdicionando os documentos no vector_store...")
    vector_store.add_documents(
        ids        = uuids,
        documents  = lc_documents, 
        embeddings = embeddings,
    )
    print("\nDocumentos carregados!")

def indexing_json_data():
    # Passo 1: Carregar dados JSON
    json_data = retrieve_processed_file_data()
    # Passo 2: Quebrar os dados em chunks
    json_chunks = break_json_data_In_chunks(json_data)
    # Passo 3: Criar modelo de embeddings
    embeddings_model = create_embeddings_model()
    # Passo 4: Gerar embeddings para os chunks
    embeddings = generate_embeddings_for_chunks(json_chunks, embeddings_model)
    # Passo 5: Converter os dados para documentos LangChain
    lc_documents = convert_json_data_to_lc_documents(json_data)
    # Passo 6: Criar o vector store (Chroma)
    vector_store = create_vector_store(embeddings_model)
    # Passo 7: Populando o vector store com documentos e embeddings
    populate_vector_store(vector_store, lc_documents, embeddings)

# Executando a função main
if __name__ == "__main__":
    indexing_json_data()
