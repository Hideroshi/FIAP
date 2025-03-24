
from fine_tuning import  create_unsloth_configurations, generate_inputs
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from transformers import TextStreamer

# Carrega o modelo base
def get_model(base_model, uc):
    model = AutoPeftModelForCausalLM.from_pretrained(
        base_model,
        load_in_4bit = uc["load_in_4bit"],
    )
    return model

def get_tokenizer(base_model):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    return tokenizer

def get_text_streamer(tokenizer):
    text_streamer = TextStreamer(tokenizer, skip_prompt = True)
    return text_streamer

def get_outputs(model, inputs, text_streamer):
    # Gera resposta
    print("\nResposta do modelo:")
    outputs = model.generate(
        **inputs,
        streamer       = text_streamer,
        max_new_tokens = 128,  # Máximo de tokens na resposta
        temperature    = 0.1,  # Controla aleatoriedade (0.0 a 1.0)
        do_sample      = True, # Usa amostragem probabilística
        use_cache      = True  # Ativa o cache para melhorar a velocidade de geração
    )
    return outputs

    # Decodifica os tokens em texto
def decode_responses(tokenizer, outputs):
    responses = tokenizer.batch_decode(outputs)
    return responses

def ask_the_model(base_model, uc, product):
    model = get_model(base_model, uc)
    tokenizer = get_tokenizer(base_model)
    inputs = generate_inputs(tokenizer, product)
    text_streamer = get_text_streamer(tokenizer)
    outputs = get_outputs(model, inputs, text_streamer)
    print("Query: ", product)
    decode_responses(tokenizer, outputs)

def ask_questions_to_the_model(base_model, uc):
    model = get_model(base_model, uc)
    tokenizer = get_tokenizer(base_model)
    products = ["Girls Ballet Tutu Neon Blue",
                 "Mog's Kittens",
                 "The Prophet",
                 "The Book of Revelation"]
    for product in products:
        inputs = generate_inputs(tokenizer, product)
        text_streamer = get_text_streamer(tokenizer)
        print("\nQuery: ", product)
        outputs = get_outputs(model, inputs, text_streamer)
        decode_responses(tokenizer, outputs)

def ask():
    uc = create_unsloth_configurations()
    base_trained_model = "./" + uc["lora_model"]

    #ask_the_model(base_trained_model, uc, "Passenger to Frankfurt")
    # ask_the_model(base_trained_model, uc, "Mog's Kittens")
    # ask_the_model(base_trained_model, uc, "The Prophet")
    # ask_the_model(base_trained_model, uc, "The Book of Revelation")

    print("\n###############################################################################")
    print(f"# Perguntando ao modelo treinado: {base_trained_model}")
    print("###############################################################################\n")
    ask_questions_to_the_model(base_trained_model, uc)

# Executando a função main
if __name__ == "__main__":
    ask()
