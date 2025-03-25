from fine_tuning import create_unsloth_configurations
from unsloth import FastLanguageModel

def load_model_and_tokenizer():
    uc = create_unsloth_configurations()

    # Carregar o modelo e o tokenizador
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = uc["model"],
        max_seq_length = uc["max_seq_length"],
        dtype          = uc["dtype"],
        load_in_4bit   = True,  # Quantização em 4 bits
        device_map     = {"": "cuda:0"}  # Habilita a utilização da GPU
    )
    # Prepara o modelo para inferência
    FastLanguageModel.for_inference(model)
    return model, tokenizer

# Definindo o template do Alpaca Prompt
# alpaca_prompt = create_alpaca_prompt()
alpaca_prompt = """
Below is an instruction that describes a task, paired with an input that
provides further context. Write a response that appropriately completes
the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""

# Função para gerar o prompt com base no produto
def generate_alpaca_prompt(instruction, product):
    return alpaca_prompt.format(instruction, product, "")

def query_model(model, tokenizer, product):
    # Gerar o prompt
    instruction = "GET THE DESCRIPTION OF THIS PRODUCT"
    prompt = generate_alpaca_prompt(instruction, product)

    # Tokenizar o prompt
    inputs = tokenizer(prompt, return_tensors="pt", padding=True,
                       truncation=True, max_length=2048).to("cuda")

    # Gerar a resposta usando o modelo
    outputs = model.generate(
        **inputs,
        max_new_tokens = 512,  # Máximo de tokens na resposta
        temperature    = 0.3,  # Controla a aleatoriedade da resposta
        do_sample      = True,  # Usar amostragem probabilística
        top_p          = 0.95,  # Nucleus sampling
        pad_token_id   = tokenizer.eos_token_id  # Definir o token de preenchimento
    )

    # Decodificar a resposta
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def extract_response(text):
    start_marker = "### Response:"

    # Encontra a posição do marcador "### Response:"
    start_index = text.find(start_marker)

    if start_index != -1:
        # Ajusta o índice para começar após o marcador
        start_index += len(start_marker)
        # Extrai o conteúdo após o marcador
        response = text[start_index:].strip()
        return response
    else:
        return "Resposta não encontrada."

def ask_the_foundation_model():
    model, tokenizer = load_model_and_tokenizer()

    products = ["Girls Ballet Tutu Neon Blue",
                "Mog's Kittens",
                "The Prophet",
                "The Book of Revelation"]

    for product in products:
        # Gerar a resposta
        response = query_model(model, tokenizer, product)
        print("\nProduto:", product)
        print("Descrição:", extract_response(response))

# Executando a função main
if __name__ == "__main__":
    ask_the_foundation_model()
