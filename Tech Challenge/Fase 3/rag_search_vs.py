from rag_indexing import create_embeddings_model, create_vector_store 

def search(vector_store, embeddings_model, product):
    print("\nRealizando consultas na vector_store")
    return vector_store.similarity_search_by_vector(
        embedding=embeddings_model.embed_query(product), k=3
    )

def print_results(results):
    for doc in results:
        print(f"\n* {doc.page_content} [{doc.metadata}]")

def search(vector_store, embeddings_model, product):
    print("\nRealizando consultas na vector_store")
    return vector_store.similarity_search_by_vector(
        embedding=embeddings_model.embed_query(product), k=3
    )

def evaluate_indexing_process():
    embeddings_model = create_embeddings_model()
    vector_store = create_vector_store(embeddings_model)
    results = search(vector_store, embeddings_model, "Girls Ballet Tutu Neon Pink")
    print_results(results)

# Executando a função principal
if __name__ == "__main__":
    evaluate_indexing_process()
