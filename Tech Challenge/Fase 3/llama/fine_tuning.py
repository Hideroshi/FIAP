import json
import sys
import torch
import triton
import transformers
from datasets import load_dataset
from pprint import pprint
from tqdm import tqdm
from transformers import TrainingArguments, TextStreamer
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

def get_gpu_stats():
    return torch.cuda.get_device_properties(0)

def get_start_gpu_memory():
    return round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)

def get_max_memory():
    return round(get_gpu_stats().total_memory / 1024 / 1024 / 1024, 3)

def display_tool_version():
    """
    Displays every tool´s version used during the fine-tuning.
    """
    print("\n###############################################################################")
    print("# Informações do Ambiente de Execução")
    print("###############################################################################\n")
    print(f"Versão do Python      : {sys.version}")
    print(f"Versão do PyTorch     : {torch.__version__}")
    print(f"Versão do Triton      : {triton.__version__}")
    print(f"Versão do Transformers: {transformers.__version__}")
    print(f"CUDA disponível       : {torch.cuda.is_available()}")
    print(f"Versão do CUDA        : {torch.version.cuda}")
    print(f"Versão do cuDNN       : {torch.backends.cudnn.version()}")
    print(f"Números de GPUs       : {torch.cuda.device_count()}")
    print(f"Nome da GPU 0         : {get_gpu_stats().name}")
    print(f"Memória Máxima        : {get_max_memory()} GB")
    print(f"Memória reservada     : {get_start_gpu_memory()} GB\n")

def create_unsloth_configurations(model = 2):
    """
    Creates unsloth configuration object ``unsloth_config`` using thespecified
    ``model`` (2 = "unsloth/llama-3-8b-bnb-4bit").
    """
    print("\n###############################################################################")
    print("# Criando Configurações do Unsloth")
    print("###############################################################################\n")
    torch_dtype = torch.bfloat16 # Formato de precisão reduzida para economia de memória
                                 # bfloat16 é bom para treinar em GPUs modernas
    # Condicional para verificar a compatibilidade com bfloat16
    if not is_bfloat16_supported():
        print("Aviso: bfloat16 não é suportado, usando fp16.")
        torch_dtype = torch.float16  # Ajusta para fp16 se bfloat16 não for suportado

    # Lista de modelos pré-treinados otimizados para 4 bits
    fourbit_models = [
        # Modelos Mistral (7B parâmetros)
        "unsloth/mistral-7b-v0.3-bnb-4bit",          # Modelo base
        "unsloth/mistral-7b-instruct-v0.3-bnb-4bit", # Versão para instruções
        # Modelos LLaMA-3
        "unsloth/llama-3-8b-bnb-4bit",               # 8B parâmetros
        "unsloth/llama-3-8b-Instruct-bnb-4bit",      # Versão instrução
        "unsloth/llama-3-70b-bnb-4bit",              # 70B parâmetros
        # Modelos Phi (menores e mais rápidos)
        "unsloth/Phi-3-mini-4k-instruct",            # Versão mini
        "unsloth/Phi-3-medium-4k-instruct",          # Versão média
        # Outros modelos
        "unsloth/mistral-7b-bnb-4bit",              # Mistral alternativo
        "unsloth/gemma-7b-bnb-4bit",                # Modelo Gemma do Google
    ]

    chosen_model = fourbit_models[model]
    unsloth_config = {
        # Tamanho máximo da sequência de entrada
        "max_seq_length"      :  8192,               # Máximo de tokens que o modelo pode processar de uma vez
        # Tipo de dados para os pesos do modelo
        "dtype"               : torch_dtype,         # Formato de precisão reduzida para economia de memória
                                                     # bfloat16 é bom para treinar em GPUs modernas
        # Habilita quantização em 4 bits
        "load_in_4bit"        : True,                # Reduz uso de memória mantendo boa performance
        "model"               : chosen_model,
        "lora_model"          : "lora_model_" + chosen_model.split("/")[-1],
        "attn_implementation" : "flash_attention_2", # Usa Flash Attention 2
        "device_map"          : "auto",              # Otimiza distribuição na GPU
    }
    print("Configuração realizada:")
    pprint(unsloth_config)
    return unsloth_config

def get_pretrained_model_and_tokenizer(uc):
    """
    Gets pretrained ``model`` and ``tokenizer``.
    """
    ###############################################################################
    # Carregamento e Configuração para o Unsloth
    ###############################################################################
    print("\n###############################################################################")
    print("# Carregando o modelo e tokenizador para o modelo...")
    print("###############################################################################\n")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name          = uc["model"],               # Usa LLaMA-3 8B (índice 2 da lista)
        max_seq_length      = uc["max_seq_length"],      # Tamanho máximo  dos tokens
        dtype               = uc["dtype"],               # Tipo de dados (bfloat16)
        load_in_4bit        = uc["load_in_4bit"],        # Quantização em 4 bits
        attn_implementation = uc["attn_implementation"], # Usa Flash Attention 2
        device_map          = uc["device_map"],          # Otimiza distribuição na GPU
    )
    print(f"\nModelo {uc["model"]} e tokenizador carregados e com sucesso!")
    return model, tokenizer

def get_peft_model(model):
    """
    Gets the ``model``, applying LoRA configuration to it, returning it
    ready to beg fine-tuned.
    """
    print("\n###############################################################################")
    print("# Configurando o modelo para fine-tuning com LoRA (Low Rank Adaptation)...")
    print("###############################################################################\n")
    peft_model = FastLanguageModel.get_peft_model(
        # Aplica LoRA para fine-tuning eficiente
        model,                                  # Modelo base carregado
        r                          = 32,        # Controla o processo de finetuning (v: +rápido,-memória; ^:+acurácia, +memória, +overfiting)
        # Camadas que serão adaptadas
        target_modules             = [
            "q_proj",                           # Query projection
            "k_proj",                           # Key projection
            "v_proj",                           # Value projection
            "o_proj",                           # Output projection
            "gate_proj",                        # Gate projection
            "up_proj",                          # Upscaling projection
            "down_proj",                        # Downscaling projection
        ],
        # Configura parâmetros de treinamento
        lora_alpha                 = 32,        # Escala de adaptação
        lora_dropout               = 0,         # Sem dropout (Não previne overfitting.)
        bias                       = "none",    # Não treina bias
        # Otimiza uso de memória
        use_gradient_checkpointing = "unsloth", # Economia de memória
        random_state               = 3407,      # Semente aleatória
        use_rslora                 = False,     # Sem RS-LoRA
        loftq_config               = None,      # Sem LoftQ
    )

    # Se o modelo suportar gradient_checkpointing, habilite para salvar memória
    if hasattr(peft_model, 'gradient_checkpointing_enable'):
        peft_model.gradient_checkpointing_enable()

    print("\nModelo configurado para fine-tuning com LoRA (Low Rank Adaptation)")
    return peft_model

def create_alpaca_prompt():
    """
    Creates the Alpaca Prompt
    """
    alpaca_prompt = """Below is an instruction that describes a task, paired 
    with an input that provides further context. Write a response that 
    appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""
    return alpaca_prompt

def count_records(file):
   """
   Counts the number of lines of a given ``file``.
   """
   total_lines = sum(1 for _ in open(file, 'r', encoding='utf-8'))
   return total_lines

def create_train_dataset(processed_file, total_lines):
    """
    Creates the train dataset ``processed_dataset``.
    """
    print("\n###############################################################################")
    print("# Criando o Conjunto de Dados para Treinamento do Modelo...")
    print("###############################################################################\n")
    processed_dataset = []
    with open(processed_file, 'r', encoding='utf-8') as file:
        for line in tqdm(file, total=total_lines, desc="Processando linhas"):
            # Remove espaços em branco e quebras de linha
            line = line.strip()
            if not line or line == "[" or line == "]":  # Pula linhas vazias, [, ]
                continue
            # Remove vírgula no final da linha se existir
            if line.endswith(','):
                line = line[:-1]

            # Converte a linha em objeto JSON
            item = json.loads(line)

            # Extrai e limpa os campos
            product = str(item.get('product', '')).strip()
            description = str(item.get('description', '')).strip()

            # Só adiciona se ambos os campos não estiverem vazios
            if product and description:
                processed_dataset.append({
                    'instruction': "GET THE DESCRIPTION OF THIS PRODUCT",
                    'input': product,
                    'output': description
                })
    return processed_dataset


def format_dataset_into_model_input(processed_dataset, dataset_file):
    """
    Formats ``processed_dataset`` into ``dataset_file``.
    """
    print("\n###############################################################################")
    print("# Gerando o Arquivo com o Conjunto de Dados para Treinamento do Modelo...")
    print("###############################################################################\n")
    # Inicializando as listas para armazenar os dados
    instructions = []
    inputs = []
    outputs = []
    # Salvar o novo arquivo dataset
    with open(dataset_file, 'w') as output_file:
        for item in tqdm(processed_dataset, desc="Salvando registros"):
            # Cria o registro no formato esperado
            instructions.append("GET THE DESCRIPTION OF THIS PRODUCT")
            inputs.append(item["input"])
            outputs.append(item["output"])

        # Criando o dicionário final
        formatted_data = {
            "instruction": instructions,
            "input": inputs,
            "output": outputs
        }
        # Salva uma linha por registro
        json.dump(formatted_data, output_file, indent = 1)
    print(f"\nDataset salvo em                             : {dataset_file}")


def load_dataset_from(dataset_file):
    """
    Loads ``dataset_file`` into ``dataset``.
    """
    print("\n###############################################################################")
    print("# Gerando o Dataset para Treinamento do Modelo...")
    print("###############################################################################\n")
    dataset = load_dataset("json", data_files=dataset_file, split = "train")
    print("\nDataset geardo com sucesso:", dataset)
    return dataset

def map_dataset_into_prompt_dataset(tokenizer, dataset):
    """
    Maps ``dataset`` to ``prompt_dataset``.
    """
    def formatting_prompts_func(examples):
        texts = []
        for i in range(len(examples['instruction'])):
            text = create_alpaca_prompt().format(
                examples['instruction'][i],
                examples['input'][i],
                examples['output'][i]) + tokenizer.eos_token
            texts.append(text)
        return { "text" : texts }

    print("\n###############################################################################")
    print("# Convertendo o Dataset para Treinamento do Modelo em Prompts...")
    print("###############################################################################\n")
    prompt_dataset = dataset.map(
        formatting_prompts_func,          # Função que será aplicada a cada batch
        batched    = True,                # Processa em batches para maior eficiência
        batch_size = 1024,                # Define tamanho do batch
        num_proc   = 1,                   # Nnúmero de processos para paralelização
        desc       = "Formatando prompts" # Descrição para a barra de progresso
    )
    print("\nDataset convertido com sucesso:", prompt_dataset)
    return prompt_dataset

def get_configured_stftrainer(uc, peft_model, prompt_dataset, tokenizer):
    """
    Gets ``trainer`` properly configured.
    """
    print("\n###############################################################################")
    print("# Configurando o Treinador SFT (Supervised Fine-Tuning)...")
    print("###############################################################################\n")
    training_arguments = TrainingArguments(
        # Tamanho dos batches e gradientes
        per_device_train_batch_size = 2,                 # Exemplos por batch na GPU
        gradient_accumulation_steps = 4,                 # Acumula gradientes antes de atualizar
        # Etapas de treinamento
        warmup_steps                = 5,                 # Passos de aquecimento
        max_steps                   = 60,                # Total de passos de treinamento (teste)
        #num_train_epochs            = 2,                 # Configuracao de treinamento real
        # Taxa de aprendizado e otimização
        learning_rate               = 2e-4,              # Taxa de aprendizado
        weight_decay                = 0.01,              # Regularização L2
        lr_scheduler_type           = "linear",          # Decaimento linear da taxa
        # Precisão e performance
        fp16 = not is_bfloat16_supported(),              # Usa FP16 se bfloat16 não disponível
        bf16 = is_bfloat16_supported(),                  # Usa bfloat16 se disponível
        gradient_checkpointing      = True,              # Ativa gradient checkpointing
        optim                       = "adamw_8bit",      # Otimizador em 8 bits
        max_grad_norm               = 0.3,               # Limita gradientes
        # Logging e saída
        logging_steps               = 1,                 # Frequência de log
        output_dir                  = "trainer_outputs", # Diretório de saída
        save_steps                  = 1000,              # Frequência de salvamento do modelo
        # Reprodutibilidade
        seed                        = 3407,              # Semente aleatória
    )

    trainer = SFTTrainer(
        # Configurações básicas
        model              = peft_model,     # Modelo LoRA configurado
        tokenizer          = tokenizer,      # Tokenizador do modelo
        train_dataset      = prompt_dataset, # Dataset formatado
        dataset_text_field = "text",         # Campo que contém o texto
        max_seq_length     = uc["max_seq_length"], # Tamanho máximo da sequência
        dataset_num_proc   = 1,              # Número de processos para processamento
        packing            = False,          # Desativa empacotamento de sequências
        # Argumentos de treinamento
        args = training_arguments,
    )
    print("\nTreinador SFT configurado com sucesso!")
    return trainer

def train_trainer(trainer):
    """
    Trains ``trainer`` properly.
    """
    print("\n###############################################################################")
    print("# Treinando o Treinador SFT (Supervised Fine-Tuning)...")
    print("###############################################################################\n")
    trainer_stats = trainer.train()

    # Análise das estatísticas de treinamento
    print("\nEstatísticas do Treinamento")
    print("===========================")
    print(f"Tempo total de treinamento       : {trainer_stats.metrics['train_runtime']:.2f} segundos ({round(trainer_stats.metrics['train_runtime']/60, 2)} min)")
    print(f"Velocidade de processamento      : {trainer_stats.metrics['train_samples_per_second']:.2f} samples/s")
    print(f"Loss final do treinamento        : {trainer_stats.metrics['train_loss']:.4f}")
    # Estatísticas de memória
    max_memory = get_max_memory()
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - get_start_gpu_memory(), 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print("\nEstatísticas de Memória GPU")
    print("============================")
    print(f"Pico de memória reservada        : {used_memory:.3f} GB")
    print(f"Pico de memória para treinamento : {used_memory_for_lora:.3f} GB")
    print(f"Percentual da memória máxima     : {used_percentage:.3f}%")
    print(f"Percentual usado no treinamento  : {lora_percentage:.3f}%")
    if used_percentage > 100:
        print("\nAVISO: O uso de memória ultrapassou o limite máximo recomendado!")
    print("\nTreinamento concluido com sucesso!")

def prepare_model_for_inference(peft_model):
    """
    Prepares ``peft_model`` for inference.
    """
    print("\n###############################################################################")
    print("# Preparando o modelo para fazer inferência (previsões)...")
    print("###############################################################################\n")
    # Prepara o modelo para inferência
    FastLanguageModel.for_inference(peft_model)
    print("\nModelo preparado para inferência!")
    return peft_model

def generate_inputs(tokenizer, product):
    alpaca_prompt = create_alpaca_prompt()
    # Tokeniza e move para GPU
    return tokenizer(
        # Lista com os prompts
        [
            # Formata o prompt usando template Alpaca
            alpaca_prompt.format(
                "GET THE DESCRIPTION OF THIS PRODUCT", # Instrução
                product,                                 # Produto
                "",                                      # Resposta vazia
            )
        ],
        padding        = True,
        truncation     = True,
        return_tensors = "pt"  # Retorna tensores PyTorch
    ).to("cuda")               # Move para GPU

def prepare_product_for_tokenization(tokenizer, product):
    """
    Prepares ``product`` for tokenization.
    """
    print("\n###############################################################################")
    print(f"# Preparando '{product}' para ser tokenizado e passar por inferência...")
    print("###############################################################################")
    # Tokeniza e move para GPU
    inputs = generate_inputs(tokenizer, product)
    print("\nTokenização realizada com sucesso!")
    return inputs

def query_model(peft_model, inputs, tokenizer, temperature = 0.2):
    """
    Query ``peft_model`` with ``inputs`` for a ``response``.
    """
    print("\n###############################################################################")
    print("# Realizando pergunta para o modelo...")
    print("###############################################################################")
    # Gera resposta
    outputs = peft_model.generate(
        **inputs,
        max_new_tokens = 128,  # Máximo de tokens na resposta
        temperature    = temperature,  # Controla aleatoriedade (0.0 a 1.0)
        do_sample      = True, # Usa amostragem probabilística
        use_cache      = True  # Ativa o cache para melhorar a velocidade de geração
    )

    # Decodifica os tokens em texto
    response = tokenizer.batch_decode(outputs)
    print("\nResposta obtida com sucesso!")
    return response

def extract_response_from(model_output):
    """
    Extract model´s response ``model_output``.
    """
    # Localizar a posição do início e do final da resposta
    start_marker = "### Response:"
    end_marker = "<|end_of_text|>"

    # Verificar se os marcadores existem na string
    if start_marker in model_output and end_marker in model_output:
        # Extrair a parte entre os dois marcadores
        start_index = model_output.index(start_marker) + len(start_marker)
        end_index = model_output.index(end_marker)

        # Extrair o conteúdo entre os índices
        response = model_output[start_index:end_index].strip()
        return response
    else:
        return "Resposta não encontrada."

def create_text_streamer(tokenizer):
    # Configuração mais detalhada
    text_streamer = TextStreamer(
        tokenizer,
        skip_prompt         = True, # Não mostra o prompt
        skip_special_tokens = True  # Não mostra tokens especiais
    )
    return text_streamer

def query_model_stream(peft_model, inputs, tokenizer):
    """
    Query ``peft_model`` with ``inputs`` for a streamming ``response``.
    """
    text_streamer = create_text_streamer(tokenizer)

    # Geração com mais parâmetros
    outputs = peft_model.generate(
        **inputs,
        streamer       = text_streamer,
        max_new_tokens = 128,  # Máximo de tokens na resposta
        temperature    = 0.2,  # Controla aleatoriedade (0.0 a 1.0)
        do_sample      = True, # Usa amostragem probabilística
        top_p          = 0.95, # Nucleus sampling
        use_cache      = True, # Ativa o cache para melhorar a velocidade de geração
    )
    # Decodifica os tokens em texto
    response = tokenizer.batch_decode(outputs)
    print("\nResposta obtida com sucesso!")
    return response

def save_all(uc, peft_model, tokenizer):
    """
    Saves ``uc``, ``peft_model`` ``tokenizer``.
    """
    print("\n###############################################################################")
    print(f"# Salvando o Modelo Treinado em {uc["lora_model"]}...")
    print("###############################################################################")
    peft_model.save_pretrained(uc["lora_model"]) #This ONLY saves the LoRA adapters, and not the full peft_model.
    tokenizer.save_pretrained(uc["lora_model"])
    peft_model.save_pretrained_merged(
        uc["lora_model"],             # Diretório onde o modelo será salvo
        tokenizer,                    # Tokenizer usado com o modelo
        save_method = "merged_16bit", # Método de salvamento em precisão de 16 bits
    )
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIM DO FINE TUNING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

def fine_tuning():
    # Exibe informações sobre as versões das ferramentas e a configuração do ambiente de execução
    display_tool_version()
    # Cria a configuração do modelo Unsloth com base no modelo especificado
    uc = create_unsloth_configurations()
    # Carrega o modelo pré-treinado e o tokenizador utilizando as configurações criadas
    model, tokenizer = get_pretrained_model_and_tokenizer(uc)
    # Aplica LoRA ao modelo, configurando-o para fine-tuning eficiente
    peft_model = get_peft_model(model)
    # Conta o número total de registros no arquivo processado (usado para treinamento)
    total_lines = count_records(processed_file)
    # Cria o dataset de treinamento a partir do arquivo processado
    train_dataset = create_train_dataset(processed_file, total_lines)
    # Formata o dataset de treinamento para o formato necessário para o modelo
    format_dataset_into_model_input(train_dataset, dataset_file)
    # Carrega o dataset do arquivo JSON para um formato compatível com Hugging Face
    dataset = load_dataset_from(dataset_file)
    # Mapeia o dataset para um formato de prompts, onde os dados são convertidos para o formato esperado pelo modelo
    prompt_dataset = map_dataset_into_prompt_dataset(tokenizer, dataset)
    # Configura o treinador para fine-tuning supervisionado (SFT) com os parâmetros apropriados
    trainer = get_configured_stftrainer(uc, peft_model, prompt_dataset, tokenizer)
    # Inicia o treinamento do modelo usando o trainer configurado
    train_trainer(trainer)

    # Prepara o modelo fine-tunado para inferência (geração de respostas)
    peft_model = prepare_model_for_inference(peft_model)

    # Prepara o prompt para o produto "Mod's Kittens" e tokeniza
    inputs = prepare_product_for_tokenization(tokenizer, "Mog's Kittens")
    # Realiza a consulta no modelo usando o prompt tokenizado, gerando uma resposta
    responses = query_model(peft_model, inputs, tokenizer)
    print(f"\nResposta do modelo: [{extract_response_from(responses[0])}]")

    # Prepara o prompt para outro produto "Mog's Kittens" e tokeniza
    inputs = prepare_product_for_tokenization(tokenizer,"Mog's Kittens")
    # Realiza uma consulta em fluxo (streaming) no modelo usando o prompt tokenizado
    query_model_stream(peft_model, inputs, tokenizer)

    # Salva o modelo fine-tunado e o tokenizador no diretório configurado, incluindo LoRA
    save_all(uc, peft_model, tokenizer)

processed_file = "trn_processed.json"
dataset_file   = "trn_processed_dataset.json"

# Executando a função main
if __name__ == "__main__":
    fine_tuning()