from rag_indexing import create_embeddings_model, create_vector_store
from fine_tuning import (create_unsloth_configurations,
                         get_pretrained_model_and_tokenizer,
                         prepare_model_for_inference)
import torch

alpaca_prompt = """
Below is an instruction that describes a task, paired with an input that 
provides further context. Write a response that appropriately completes 
the request.

### Instruction:
{} {}

### Input:
{}

### Response:
{}
"""

def extract_response(answer):
    """
    Extrai e retorna o texto após "### Response:" da resposta gerada.
    
    Args:
    - answer (str): A resposta gerada, que contém o texto após "### Response:".
    
    Returns:
    - response_text (str): O texto após "### Response:".
    """
    # Procurando pela linha "### Response:" e extraindo o conteúdo após ela
    response_start = "### Response:"
    if response_start in answer:
        # Encontra o índice onde "### Response:" começa
        start_index = answer.index(response_start) + len(response_start)
        # Extrai o texto após "### Response:"
        response_text = answer[start_index:].strip()
        return response_text
    else:
        return "Nenhuma resposta encontrada."

def extract_response(answer):
    """
    Extrai e retorna o texto após "### Response:" da resposta gerada.

    Args:
    - answer (str): A resposta gerada, que contém o texto após "### Response:".

    Returns:
    - response_text (str): O texto após "### Response:".
    """
    # Procurando pela linha "### Response:" e extraindo o conteúdo após ela
    response_start = "### Response:"
    if response_start in answer:
        # Encontra o índice onde "### Response:" começa
        start_index = answer.index(response_start) + len(response_start)
        # Extrai o texto após "### Response:"
        response_text = answer[start_index:].strip()
        return response_text
    else:
        return "Nenhuma resposta encontrada."

def create_prompt(product, context):
    return alpaca_prompt.format(
        "GET THE DESCRIPTION OF THIS PRODUCT:",
        product,
        context,
        ""
    )
def tokenize_inputs(tokenizer, prompt, uc):
    return tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=uc["max_seq_length"]
    )

def generate_outputs(peft_model, inputs, tokenizer):
    with torch.no_grad():
        outputs = peft_model.generate(
            **inputs,
            max_length=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )
    return outputs

def ask_the_model(retriever, product, peft_model, tokenizer, uc):
    retrieved_docs = retriever.get_relevant_documents(product)
    context = "\n".join(
        [doc.page_content if hasattr(doc, "page_content") else doc for doc in
         retrieved_docs])
    prompt = create_prompt(product, context)
    inputs = tokenize_inputs(tokenizer, prompt, uc)
    inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
    outputs = generate_outputs(peft_model, inputs, tokenizer)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

def ask_the_model_using_rag():
    embeddings_model = create_embeddings_model()
    vector_store = create_vector_store(embeddings_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    uc = create_unsloth_configurations()
    model, tokenizer = get_pretrained_model_and_tokenizer(uc)
    peft_model = prepare_model_for_inference(model)

    product = "Girls Ballet Tutu Neon Blue"
    answer = ask_the_model(retriever, product, peft_model, tokenizer, uc)
    print("\n\nResposta gerada:")
    print(answer)

    product = "Mog's Kittens"
    answer = ask_the_model(retriever, product, peft_model, tokenizer, uc)
    print("\n\nResposta gerada:")
    print(answer)

if __name__ == "__main__":
    ask_the_model_using_rag()
