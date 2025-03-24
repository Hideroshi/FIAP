# Fine-Tuning de LLM com Unsloth e LoRA

## Visão Geral do Processo

Este documento descreve o processo de fine-tuning realizado em um modelo de linguagem (LLM). O código ajusta um modelo pré-treinado (como Llama-3-8B) para gerar descrições de produtos com base em instruções estruturadas, uitilizando: 

- **Unsloth** para otimização de desempenho
- **LoRA** para adaptação eficiente
- **SFT** (Supervised Fine-Tuning)

## Fluxo Principal
1. Configuração do ambiente
2. Carregamento do modelo e tokenizador
3. Aplicação de LoRA
4. Preparação do dataset
5. Treinamento supervisionado
6. Inferência e salvamento

---

# 1. Preparação do Ambiente

Este trabalho foi planejado para rodar no **Windows**. Portanto, é necessário atentar-se à instalação correta das bibliotecas e dependências.

### Configuração do Ambiente:

( Configuração utilizada nos testes)
- **Sistema Operacional:** Windows 11
- **Python:** 3.12.6
- **GPU:** RTX 4060
- **CUDA:** 12.6
- **cuDNN:** 9.8

### Passos de Instalação (executar na ordem):

1. [CUDA Toolkit 12.6](https://developer.nvidia.com/cuda-12-6-3-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
2. [cuDNN 9.8.0 Downloads](https://developer.nvidia.com/cudnn-downloads)
3. [triton-windows](https://github.com/woct0rdho/triton-windows)
4. [PyTorch](https://pytorch.org/get-started/locally/)
5. [Unsloth](https://docs.unsloth.ai/get-started/installing-+-updating/windows-installation)

### Observação sobre o Unsloth:

A versão `2025.2.X` apresentou problemas. Foi utilizada a versão `2025.1.5` neste teste:

```bash
pip install unsloth==2025.1.5
```

A versão de março foi testada em outra máquina e funciona também:

```bash
pip install --no-deps "unsloth>=2025.3.8" "unsloth_zoo>=2025.3.7" --upgrade --force-reinstall
```

---

## 2. Processamento dos Dados

### **Fonte dos Dados**

Os dados utilizados foram obtidos do [Amazon Titles Reasoning](https://huggingface.co/datasets/rickwalking/amazon-titles-reasoning).

### **Estrutura dos Dados**

Trabalhamos com o arquivo `trn.json`, que contém os dados de treino no formato:

```json 
{"uid": "0001360000", "title": "Mog's Kittens", "content": "Judith Kerr&#8217;s best&#8211;selling adventures of that endearing (and exasperating) cat Mog have entertained children for more than 30 years. Now, even infants and toddlers can enjoy meeting this loveable feline. These sturdy little board books&#8212;with their bright, simple pictures, easy text, and hand&#8211;friendly formats&#8212;are just the thing to delight the very young. Ages 6 months&#8211;2 years.", "target_ind": [146, 147, 148, 149, 495], "target_rel": [1.0, 1.0, 1.0, 1.0, 1.0]}
{"uid": "0000031895", "title": "Girls Ballet Tutu Neon Blue", "content": "Dance tutu for girls ages 2-8 years. Perfect for dance practice, recitals and performances, costumes or just for fun!", "target_ind": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 31, 33, 42, 46, 54, 58, 111, 113, 125, 126, 159, 163, 202, 203, 204, 205, 206, 207], "target_rel": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}
```

O arquivo `trn.json` contém **2.248.619 registros**. Para o fine-tuning, utilizamos apenas os campos `title` e `content` e os demais campos serão descartados.

### **Processamento dos Dados**

O arquivo `process_data.py` realiza o processamento dos dados: 
  1. **Filtragem:** Registros com `content` vazio ou <100 caracteres são descartados.
  2. **Geração do dataset final:** Criado o arquivo `trn_processed.json` com **1.216.560 registros válidos**.

Vamos descartar os registros com `content` vazio ou com o conteúdo menor do que 100 caracteres.
Com isto, geramos o arquivo `trn_processed.json`, com 1.216.560 registros aptos 
a serem usados no treinamento do modelo.

### **Formato do Arquivo Processado:**

```json 
[
 {"product": "Mogs Kittens", "description": "Judith Kerr8217s best8211selling adventures of that endearing and exasperating cat Mog have entertained children for more than 30 years Now even infants and toddlers can enjoy meeting this loveable feline These sturdy little board books8212with their bright simple pictures easy text and hand8211friendly formats8212are just the thing to delight the very young Ages 6 months82112 years"},
 {"product": "Girls Ballet Tutu Neon Blue", "description": "Dance tutu for girls ages 28 years Perfect for dance practice recitals and performances costumes or just for fun"},
]
```

## Exemplo de execução:
``` 
(.venv) PS C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama> python .\process_data.py  
  
Lendo os registros do arquivo              : 100%|███████████████| 2248619/2248619 [00:35<00:00, 62903.00it/s]  
Processando registros lidos                : 100%|███████████████| 2248619/2248619 [00:14<00:00, 155615.25it/s]  
Salvando registros processado em arquivo   : 100%|███████████████| 1216560/1216560 [00:10<00:00, 117362.03it/s]  
Total de registros no arquivo original     : 2,248,619  
Total de registros processados (não vazios): 1,216,560  
Arquivo processado salvo em                : trn_processed.json  
```
---

# 3. Foundation Model (Modelo Base)

### **Modelo Escolhido:**

> **`unsloth/llama-3-8b-bnb-4bit`** - Implementação otimizada do LLaMA com:

- **Quantização em 4 bits**
  - A quantização em 4 bits reduz o consumo de memória do modelo ao armazenar pesos de redes neurais com menor precisão.
  - Modelos tradicionais armazenam pesos com 32 bits (FP32) ou 16 bits (FP16), consumindo mais memória.
  - 4-bit quantization representa cada peso com apenas 4 bits, reduzindo drasticamente o uso de memória e permitindo a execução do modelo em GPUs com menos VRAM.
  - Isso melhora a eficiência computacional, mas pode resultar em leve perda de precisão
- **Compatível com Hugging Face Transformers**
  - O modelo pode ser carregado e utilizado diretamente com a biblioteca Hugging Face Transformers, que é um framework popular para LLMs.
- **Projetado para GPUs com CUDA** : CUDA (Compute Unified Device Architecture) é a plataforma de computação paralela da NVIDIA, permitindo a execução eficiente de redes neurais em GPUs.
  - Otimização para CUDA: O modelo aproveita operações aceleradas por GPU, como Flash Attention, que melhora a eficiência da memória.
  - Desempenho: Permite rodar inferências e treinamentos muito mais rápido do que em CPUs.
  - Compatibilidade: Suporta GPUs com Tensor Cores (ex: RTX 30xx, 40xx).
- **8 bilhões de parâmetros**

**Características:**

- Otimizado para eficiência em memória e desempenho
- Suporta Flash Attention e LoRA
- Ideal para fine-tuning em recursos limitados
- Adequado para RAG (Retrieval Augmented Generation)
    
📌 **Fonte:** [Hugging Face Model Card](https://huggingface.co/unsloth/llama-3-8b-bnb-4bit)

O modelo escolhido para realizar o fine tunning neste trabalho é o 
`unsloth/llama-3-8b-bnb-4bit`. Trata-se de uma implementação otimizada do modelo 
LLaMA (Large Language Model Meta AI), ajustada para ser eficiente em termos de 
memória e desempenho. Ele utiliza quantização em 4 bits, o que reduz 
significativamente o consumo de memória sem comprometer a precisão do modelo. 
Além disso, o modelo é compatível com a biblioteca Hugging Face Transformers e 
foi projetado para ser executado em GPUs com suporte a CUDA, como a NVIDIA 
RTX 4060, aproveitando tecnologias como Flash Attention e LoRA (Low-Rank 
Adaptation) para acelerar o treinamento e a inferência. Com um tamanho de 8 
bilhões de parâmetros, ele é capaz de lidar com tarefas complexas de 
processamento de linguagem natural, como geração de texto, resumo, tradução e 
respostas a perguntas. 

O objetivo principal do modelo é permitir o fine-tuning eficiente em cenários 
com recursos computacionais limitados, como laptops ou estações de trabalho com 
GPUs de médio porte. Ele foi otimizado para tarefas que exigem personalização, 
como a adaptação a domínios específicos ou a criação de sistemas de recuperação 
de informações baseados em RAG (Retrieval Augmented Generation). A combinação de 
quantização em 4 bits e adaptadores LoRA permite que o modelo seja ajustado 
rapidamente com conjuntos de dados menores, mantendo alta qualidade nas respostas 
geradas e reduzindo o tempo e os custos associados ao treinamento de modelos grandes.

# Exemplo de execução:

```
(.venv) PS C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama> python .\foundation_model.py
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
🦥 Unsloth Zoo will now patch everything to make training faster!

###############################################################################
# Criando Configurações do Unsloth
###############################################################################

Configuração realizada:
{'attn_implementation': 'flash_attention_2',
 'device_map': 'auto',
 'dtype': torch.bfloat16,
 'load_in_4bit': True,
 'lora_model': 'lora_model_llama-3-8b-bnb-4bit',
 'max_seq_length': 8192,
 'model': 'unsloth/llama-3-8b-bnb-4bit'}
==((====))==  Unsloth 2025.1.5: Fast Llama patching. Transformers: 4.49.0.
   \\   /|    GPU: NVIDIA GeForce RTX 4060 Laptop GPU. Max memory: 7.996 GB. Platform: Windows.
O^O/ \_/ \    Torch: 2.6.0+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.2.0
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post3. FA2 = False]
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\.venv\Lib\site-packages\unsloth\models\llama.py:1185: 
UserWarning: expandable_segments not supported on this platform (Triggered internally at 
C:\actions-runner\_work\pytorch\pytorch\pytorch\c10/cuda/CUDAAllocatorConfig.h:28.)
  self.register_buffer("cos_cached", emb.cos().to(dtype=dtype, device=device, non_blocking=True), persistent=False)

Produto: Girls Ballet Tutu Neon Blue
Descrição: This is a beautiful tutu. It is made of high quality material and it 
is very comfortable. It is perfect for any occasion and it is very affordable. 
It is also very durable and it will last for a long time. It is also very easy 
to maintain and it is very easy to clean. It is also very easy to wear and it 
is very easy to take off. It is also very easy to put on and it is very easy to 
take off. It is also very easy to put on and it is very easy to take off. It is 
also very easy to put on and it is very easy to take off. It is also very easy 
to put on and it is very easy to take off. It is also very easy to put on and it 
is very easy to take off. It is also very easy to put on and it is very easy to 
take off. It is also very easy to put on and it is very easy to take off. It is 
also very easy to put on and it is very easy to take off. It is also very easy 
to put on and it is very easy to take off. It is also very easy to put on and it 
is very easy to take off. It is also very easy to put on and it is very easy to 
take off. It is also very easy to put on and it is very easy to take off. It is 
also very easy to put on and it is very easy to take off. 

Produto: Mog's Kittens
Descrição: The product is a pair of socks.

Produto: The Prophet
Descrição: The Prophet is a product that helps you to find the best way to spend 
your money. It is a product that helps you to find the best way to spend your 
money. It is a product that helps you to find the best way to spend your money. 
It is a product that helps you to find the best way to spend your money. It is a 
product that helps you to find the best way to spend your money. It is a product 
that helps you to find the best way to spend your money. It is a product that 
helps you to find the best way to spend your money. It is a product that helps 
you to find the best way to spend your money. It is a product that helps you to 
find the best way to spend your money. It is a product that helps you to find 
the best way to spend your money. It is a product that helps you to find the 
best way to spend your money. 

Produto: The Book of Revelation
Descrição: The Book of Revelation is a book of the New Testament of the Bible, 
and its title originated from the first word of the text in the Koine Greek: 
apokalypsis, meaning "unveiling" or "revelation". The author describes himself 
as "John" and does not identify himself as the son of Zebedee, the apostle John. 
The text is a letter to seven churches in the Roman province of Asia, and is a 
call to the churches to remain faithful to Jesus Christ, and individual letters 
to each church, with a promise of a swift punishment for Christian communities 
that are in a state of apostasy. The Book of Revelation is the final book of the 
New Testament and occupies a central place in Christian eschatology. By tradition, 
this prophecy was revealed by its author to the apostle John on the island of 
Patmos, and from its first readers, this prophecy has been accepted as of divine 
inspiration. The author of Revelation does not identify himself, but introduces 
his work as "the revelation of Jesus Christ", which he received "by an angel" from God. 

Process finished with exit code 0
```
---

# 4. Fine-Tuning do Modelo

## Processo em Duas Etapas
1. **Treinamento do modelo base**
2. **Interrogação do modelo treinado**

O processo de fine-tuning do modelo envolve duas etapas: na primeira, realizamos o fine-tuning do modelo escolhido `("unsloth/llama-3-8b-bnb-4bit")` enquanto na segunda, interrogamos o modelo treinado `(./lora_model_llama-3-8b-bnb-4bit)`.

## Treinamento do Modelo

Nosso código para o treinamento do modelo está disponível no arquivo 
`fine_tuning.py`.

O proceso de fine-tuning é feito a partir do arquivo processado 
`trn_processed.json` na fase de preparação dos dados para treinamento.

**Principais Componentes:**
  - Uso de Unsloth e LoRA para eficiência
  - Formatação Alpaca para prompts
  - Redução para 5.000 itens (devido a restrições de tempo)

O uso do Unsloth e do LoRA foi motivado pela necessidade 
de realizar o fine-tuning de modelos grandes de forma eficiente e com menor 
consumo de recursos computacionais. O Unsloth oferece otimizações específicas 
para acelerar o treinamento, enquanto o LoRA permite ajustar apenas um pequeno 
número de parâmetros, reduzindo significativamente os requisitos de memória e 
tempo de execução, sem comprometer a qualidade do modelo treinado.

Inicialmente, a configuração do Unsloth é feita de forma a preparar o modelo 
para fine-tuning. Com a configuração pronta, modelo e tokenizador são carregados
e retornado. Depois, aplicamos os adaptadores LoRA ao mesmo modelo, Deixando-o 
pronto para o treinamento.

Neste momento, iniciamos a preparação dos dados que serão utilizados para o 
fine-tuning do modelo. Ele é preparado de forma a agrupar os produtos como 
grupo de entrada (`input`) e as descrições como grupo de saída (`output`). As 
instruções (perguntas que serão feitas ao modelo) ficam no grupo de 
instruções (`instructions`). Esta preparação é salva em arquivo 
(`trn_processed_dataset.json`) para ser utilizada mais adiante.

Além disso, o modelo foi ajustado utilizando o Alpaca, que aprimora a capacidade 
do modelo em lidar com consultas complexas, garantindo respostas mais precisas e 
relevantes, aumentando a eficiência em recuperação de informações e geração de 
textos.

Após ser carregado como um dataset, é transformado em um "prompt dataset"
para ser passado para o treinador do modelo. Então, o treinamento é realizado.
Neste momento, uma mudança de foco precisou ser feita: a quantidade de itens 
a serem treinadas (1.216.560) implicava em um tempo de treinamento que excediam 
7 dias. Com isto, o "prompt dataset" foi reduzido para 5000 item, o que permitiu
a aplicação de duas épocas de treinamento (algo em torno de 1250 iterações) 
tomando um total 2:30h para que a tarefa fosse realizada. É importante ressaltar 
que a configuração presente no código, a GPU utilizada e o conjunto de dados de 
treino influenciam diretamente no tempo de execução do treinamento. Algumas 
estatísticas são mostradas após o treino (tempo gasto, consumo de memória).

Ao finalizarmos esta etapa, preparamos o modelo treinado para a realização de 
inferências. Duas consultas são realizadas, apenas para mostrar as capacidades ### ARRUMAR
de exibição das respostas às consultas: a exibição completa, após o resultado 
retornado ou sua apresentação conforme o modelo vai gerando a resposta.

Concluído este pequeno teste, salvamos o modelo e seus adaptadores LoRA 
localmente.

O dataset foi carregado e processado de forma a ser compatível com o formato 
esperado pela biblioteca Hugging Face. Isso permite que ele seja utilizado 
diretamente em pipelines de treinamento e inferência, facilitando a integração 
com modelos de aprendizado de máquina e otimizando o fluxo de trabalho.

## **Parâmetros de Treinamento:**

### **🔹 Tamanho dos Batches e Gradientes**

| Argumento | Descrição |
|-----------|------------|
| `per_device_train_batch_size = 2` | Define o número de exemplos processados por batch em cada GPU. Um batch pequeno consome menos memória, mas pode afetar a estabilidade do treinamento. |
| `gradient_accumulation_steps = 4` | Acumula gradientes por 4 passos antes de atualizar os pesos do modelo. Isso simula um batch maior sem exigir mais memória da GPU. |

**Exemplo:** Se `batch_size = 2` e `gradient_accumulation_steps = 4`, o modelo só atualiza os pesos após processar **8 exemplos**.

---

### **🔹 Etapas de Treinamento**

| Argumento | Descrição |
|-----------|------------|
| `warmup_steps = 5` | Número de passos iniciais onde a taxa de aprendizado cresce gradualmente para evitar variações bruscas no gradiente. |
| `max_steps = 60` | Número total de passos de treinamento. Neste caso, é um teste. Para um treinamento real, pode-se definir `num_train_epochs`. |
| `#num_train_epochs = 2` | Define quantas épocas completas o dataset será percorrido durante o treinamento. |

---

### **🔹 Taxa de Aprendizado e Otimização**

| Argumento | Descrição |
|-----------|------------|
| `learning_rate = 2e-4` | Define a taxa de aprendizado do otimizador. Valores altos aceleram o aprendizado, mas podem ser instáveis. |
| `weight_decay = 0.01` | Regularização L2 para evitar overfitting, penalizando pesos muito grandes. |
| `lr_scheduler_type = "linear"` | Define o decaimento da taxa de aprendizado. O tipo `linear` reduz a taxa gradualmente até o final do treinamento. |

---

### **🔹 Precisão e Performance**

| Argumento | Descrição |
|-----------|------------|
| `fp16 = not is_bfloat16_supported()` | Usa **FP16** (16-bit floating point) se `bfloat16` não estiver disponível. FP16 economiza memória, mas pode ser instável. |
| `bf16 = is_bfloat16_supported()` | Usa **bfloat16** se a GPU suportar. BF16 é mais estável que FP16, consumindo a mesma quantidade de memória. |
| `gradient_checkpointing = True` | Ativa **gradient checkpointing**, salvando menos ativações durante o forward pass para economizar VRAM. Isso reduz o consumo de memória, mas aumenta o tempo de treinamento. |
| `optim = "adamw_8bit"` | Usa o otimizador **AdamW** em 8 bits, reduzindo o uso de memória do otimizador sem perder eficiência. |
| `max_grad_norm = 0.3` | Limita o valor máximo dos gradientes para evitar explosões no treinamento. |

---

### **🔹 Logging e Salvamento**

| Argumento | Descrição |
|-----------|------------|
| `logging_steps = 1` | Define a frequência com que métricas como **loss** são registradas. Valores menores geram logs mais frequentes. |
| `output_dir = "trainer_outputs"` | Define o diretório onde os logs e checkpoints do modelo serão salvos. |
| `save_steps = 1000` | Frequência com que o modelo é salvo durante o treinamento. Um valor muito baixo pode gerar arquivos desnecessários e ocupar espaço. |

---

### **🔹 Reprodutibilidade**

| Argumento | Descrição |
|-----------|------------|
| `seed = 3407` | Define uma semente fixa para garantir que os experimentos sejam reproduzíveis. Isso significa que, ao rodar o treinamento novamente, os resultados serão os mesmos. |

---

## **SFTTrainer**

O `SFTTrainer` (Supervised Fine-Tuning Trainer) é uma classe especializada para **fine-tuning eficiente** usando LoRA. Ele recebe os argumentos definidos acima (`args = training_arguments`) e adiciona configurações específicas.

### **🔹 Configurações Básicas**

| Argumento | Descrição |
|-----------|------------|
| `model = peft_model` | O modelo que será treinado. Neste caso, um modelo **LoRA ajustado**. |
| `tokenizer = tokenizer` | O tokenizador usado para processar os textos antes do treinamento. |
| `train_dataset = prompt_dataset` | O dataset formatado no padrão necessário para o treinamento. |
| `dataset_text_field = "text"` | Define qual campo do dataset contém o texto a ser usado no treinamento. |

---

### **🔹 Tamanho da Sequência e Processamento**

| Argumento | Descrição |
|-----------|------------|
| `max_seq_length = 8192` | Define o tamanho máximo de tokens que o modelo pode processar em uma única entrada. |
| `dataset_num_proc = 1` | Número de processos paralelos para pré-processamento do dataset. Valores maiores podem acelerar, mas exigem mais CPU. |
| `packing = False` | Define se entradas curtas devem ser concatenadas para otimizar o uso de espaço. No caso, está desativado. |

---

# Explicação dos Parâmetros de Geração de Texto

A função `.generate()` do modelo ajustado (`peft_model`) é responsável por gerar texto com base em um input processado. Cada argumento influencia diretamente **a forma como o modelo gera texto**, afetando **comprimento, aleatoriedade e eficiência**.

```python
outputs = peft_model.generate(
    **inputs,
    max_new_tokens = 128,  # Máximo de tokens na resposta
    temperature    = 0.2,  # Controla aleatoriedade (0.0 a 1.0)
    do_sample      = True, # Usa amostragem probabilística
    use_cache      = True  # Ativa cache para melhorar a velocidade de geração
)
```

| Parâmetro | O que faz? | Valores recomendados |
|-----------|-----------|---------------------|
| **`max_new_tokens`** | Define **o número máximo de novos tokens** que podem ser gerados **na resposta** | `50-200` (depende do contexto) |
| **`temperature`** | Controla o nível de **aleatoriedade** da geração de texto | `0.3` (formal) - `0.8` (criativo) |
| **`do_sample`** | Ativa **amostragem probabilística** | `False` (respostas fixas) - `True` (criatividade) |
| **`use_cache`** | Usa cache para **acelerar geração** | Sempre `True` |

**🔹Exemplo "max_new_tokens":** Se `max_new_tokens = 128`, o modelo pode **gerar até 128 tokens** depois do prompt de entrada.
  - Um valor muito **baixo** pode truncar a resposta antes que ela seja concluída.
  - Um valor muito **alto** pode gerar respostas longas e desnecessárias, consumindo mais memória e tempo de inferência

**🔹Exemplo "temperature":** Escolhemos utilizar uma temperatura de `0.2` pois queriamos algo mais fixo, visto que deve ter um comportamento mais inclinado ao `RAG`.
  - `temperature = 0.0` → **Texto mais determinístico** (o modelo escolhe sempre o token com maior probabilidade).
  - `temperature = 1.0` → **Texto mais criativo e diversificado** (o modelo escolhe tokens menos prováveis com mais frequência).

**🔹Exemplo "do_sample":** Permite que o modelo **não escolha sempre o token mais provável**.
  - Se `do_sample = False`, o modelo **sempre escolhe o token com maior probabilidade**, tornando as respostas **muito previsíveis**.
  - Se `do_sample = True`, o modelo **pode escolher tokens menos prováveis**, tornando o texto mais **variado e criativo**.

**🔹Exemplo "use_cache":** Ativa um **cache interno** para acelerar a geração de tokens.
  - Durante a geração, o modelo precisa **calcular os tokens anteriores repetidamente**.
  - Com **`use_cache = True`**, ele **armazena os tokens já processados**, evitando recomputação desnecessária.

--- 

### **Estatísticas de Treinamento:**

```
Tempo total de treinamento: 49274.93 segundos (821.25 min)
Velocidade de processamento: 0.20 samples/s
Loss final: 1.8543
Memória GPU utilizada: 9.125 GB (114.12% da capacidade)
```
📌 **Modelo treinado disponível em:** [ACMattosHE/lora_model_llama-3-8b-bnb-4bit](https://huggingface.co/ACMattosHE/lora_model_llama-3-8b-bnb-4bit/tree/main)

## Exemplo de execução:
A seguir, o log de execução deste código:

``` 
(.venv) PS C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama> python .\fine_tuning.py  
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.  
🦥 Unsloth Zoo will now patch everything to make training faster!  

###############################################################################  
# Informações do Ambiente de Execução  
###############################################################################  
  
Versão do Python      : 3.12.6 (tags/v3.12.6:a4a2d2b, Sep  6 2024, 20:11:23) [MSC v.1940 64 bit (AMD64)]  
Versão do PyTorch     : 2.6.0+cu126  
Versão do Triton      : 3.2.0  
Versão do Transformers: 4.49.0  
CUDA disponível       : True  
Versão do CUDA        : 12.6  
Versão do cuDNN       : 90501  
Números de GPUs       : 1  
Nome da GPU 0         : NVIDIA GeForce RTX 4060 Laptop GPU  
Memória Máxima        : 7.996 GB  
Memória reservada     : 0.0 GB  

###############################################################################  
# Criando Configurações do Unsloth  
###############################################################################  
  
Configuração realizada:  
{'attn_implementation': 'flash_attention_2',  
 'device_map': 'auto',  
 'dtype': torch.bfloat16,  
 'load_in_4bit': True,  
 'lora_model': 'lora_model_llama-3-8b-bnb-4bit',  
 'max_seq_length': 8192,  
 'model': 'unsloth/llama-3-8b-bnb-4bit'}  

###############################################################################  
# Carregando o modelo e tokenizador para o modelo...  
###############################################################################  
  
==((====))==  Unsloth 2025.1.5: Fast Llama patching. Transformers: 4.49.0.  
   \\   /|    GPU: NVIDIA GeForce RTX 4060 Laptop GPU. Max memory: 7.996 GB. Platform: Windows.  
O^O/ \_/ \    Torch: 2.6.0+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.2.0  
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post3. FA2 = False]  
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth  
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!  
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\.venv\Lib\site-packages\unsloth\models\llama.py:1185: UserWarning: expandable_segments not supported on this platform (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\pytorch\c10/cuda/CUDAAllocatorConfig.h:28.)  
  self.register_buffer("cos_cached", emb.cos().to(dtype=dtype, device=device, non_blocking=True), persistent=False)  
  
Modelo unsloth/llama-3-8b-bnb-4bit e tokenizador carregados e com sucesso!  
  
###############################################################################  
# Configurando o modelo para fine-tuning com LoRA (Low Rank Adaptation)...  
###############################################################################  
  
Unsloth 2025.1.5 patched 32 layers with 32 QKV layers, 32 O layers and 32 MLP layers.  
  
Modelo configurado para fine-tuning com LoRA (Low Rank Adaptation)  
  
###############################################################################  
# Criando o Conjunto de Dados para Treinamento do Modelo...  
###############################################################################  
  
Processando linhas: 100%|██████████████| 5002/5002 [00:00<00:00, 277916.39it/s]   
  
###############################################################################  
# Gerando o Arquivo com o Conjunto de Dados para Treinamento do Modelo...  
###############################################################################  
  
Salvando registros: 100%|██████████████| 5000/5000 [00:00<00:00, 5001554.97it/s]  
  
Dataset salvo em                             : trn_processed_dataset.json  
  
###############################################################################  
# Gerando o Dataset para Treinamento do Modelo...  
###############################################################################  
  
Generating train split: 5000 examples [00:00, 59244.76 examples/s]  
  
Dataset geardo com sucesso: Dataset({  
    features: ['instruction', 'input', 'output'],  
    num_rows: 5000  
})  
  
###############################################################################  
# Convertendo o Dataset para Treinamento do Modelo em Prompts...  
###############################################################################  
  
Formatando prompts: 100%|████████████████| 5000/5000 [00:00<00:00, 70249.52 examples/s]  
  
Dataset convertido com sucesso: Dataset({  
    features: ['instruction', 'input', 'output', 'text'],  
    num_rows: 5000  
})  
  
###############################################################################  
# Configurando o Treinador SFT (Supervised Fine-Tuning)...  
###############################################################################  
  
Map: 100%|██████████████████████████| 5000/5000 [00:00<00:00, 6432.07 examples/s]  
No label_names provided for model class `PeftModelForCausalLM`. Since `PeftModel` 
hides base models input arguments, if label_names is not given, label_names can't 
be set automatically within `Trainer`. Note that empty label_names list will be 
used instead.       
  
Treinador SFT configurado com sucesso!  
  
###############################################################################  
# Treinando o Treinador SFT (Supervised Fine-Tuning)...  
###############################################################################  
  
==((====))==  Unsloth - 2x faster free finetuning | Num GPUs = 1  
   \\   /|    Num examples = 5,000 | Num Epochs = 1  
O^O/ \_/ \    Batch size per device = 2 | Gradient Accumulation steps = 4  
\        /    Total batch size = 8 | Total steps = 1,250  
 "-____-"     Number of trainable parameters = 83,886,080  
{'loss': 3.2363, 'grad_norm': 1.2117664813995361, 'learning_rate': 0.00019984, 'epoch': 0.0}
{'loss': 3.1007, 'grad_norm': 0.7599466443061829, 'learning_rate': 0.00019968, 'epoch': 0.0}
{'loss': 2.8482, 'grad_norm': 1.2890647649765015, 'learning_rate': 0.00019952000000000001, 'epoch': 0.0}
{'loss': 2.6395, 'grad_norm': 0.8345205187797546, 'learning_rate': 0.00019936000000000002, 'epoch': 0.01}
{'loss': 2.6946, 'grad_norm': 0.9719563722610474, 'learning_rate': 0.00019920000000000002, 'epoch': 0.01}
{'loss': 2.6354, 'grad_norm': 1.515956997871399, 'learning_rate': 0.00019904, 'epoch': 0.01} 
{'loss': 2.4815, 'grad_norm': 0.8237305283546448, 'learning_rate': 0.00019888, 'epoch': 0.01}
{'loss': 2.6736, 'grad_norm': 0.7439571022987366, 'learning_rate': 0.00019872000000000002, 'epoch': 0.01}
{'loss': 2.415, 'grad_norm': 0.7331687211990356, 'learning_rate': 0.00019856000000000002, 'epoch': 0.01}
{'loss': 2.3698, 'grad_norm': 0.9583930373191833, 'learning_rate': 0.0001984, 'epoch': 0.02}
{'loss': 1.9416, 'grad_norm': 1.1305814981460571, 'learning_rate': 0.00019824, 'epoch': 0.02}
{'loss': 2.1291, 'grad_norm': 1.077892780303955, 'learning_rate': 0.00019808, 'epoch': 0.02} 
{'loss': 2.4154, 'grad_norm': 3.3878471851348877, 'learning_rate': 0.00019792000000000003, 'epoch': 0.02}
{'loss': 1.9818, 'grad_norm': 0.8834867477416992, 'learning_rate': 0.00019776, 'epoch': 0.02}
{'loss': 2.082, 'grad_norm': 1.0258870124816895, 'learning_rate': 0.0001976, 'epoch': 0.02}
{'loss': 2.0796, 'grad_norm': 0.9682966470718384, 'learning_rate': 0.00019744, 'epoch': 0.03}
{'loss': 2.1334, 'grad_norm': 0.7916772961616516, 'learning_rate': 0.00019728, 'epoch': 0.03}
{'loss': 2.1901, 'grad_norm': 0.6517899036407471, 'learning_rate': 0.00019712, 'epoch': 0.03}
{'loss': 2.0756, 'grad_norm': 0.8336527347564697, 'learning_rate': 0.00019696, 'epoch': 0.03} 
...........
{'loss': 1.6027, 'grad_norm': 0.6916661858558655, 'learning_rate': 1.44e-06, 'epoch': 1.99}
{'loss': 1.164, 'grad_norm': 0.7378243207931519, 'learning_rate': 1.28e-06, 'epoch': 1.99}
{'loss': 1.5617, 'grad_norm': 0.870815098285675, 'learning_rate': 1.12e-06, 'epoch': 1.99}
{'loss': 1.8097, 'grad_norm': 0.7309754490852356, 'learning_rate': 9.6e-07, 'epoch': 1.99}
{'loss': 1.8802, 'grad_norm': 0.7114237546920776, 'learning_rate': 8.000000000000001e-07, 'epoch': 1.99}
{'loss': 1.945, 'grad_norm': 0.8869221806526184, 'learning_rate': 6.4e-07, 'epoch': 1.99}
{'loss': 1.7192, 'grad_norm': 0.7447224855422974, 'learning_rate': 4.8e-07, 'epoch': 2.0}
{'loss': 1.5252, 'grad_norm': 0.905275821685791, 'learning_rate': 3.2e-07, 'epoch': 2.0}
{'loss': 1.8738, 'grad_norm': 0.8697205185890198, 'learning_rate': 1.6e-07, 'epoch': 2.0}
{'loss': 1.6066, 'grad_norm': 0.8066918253898621, 'learning_rate': 0.0, 'epoch': 2.0}
{'train_runtime': 49274.9322, 'train_samples_per_second': 0.203, 'train_steps_per_second': 0.025, 'train_loss': 1.8542551296710967, 'epoch': 2.0}
100%|███████████████████████████████████████████████████████████████| 1250/1250 [13:41:14<00:00, 39.42s/it] 

Estatísticas do Treinamento
===========================
Tempo total de treinamento       : 49274.93 segundos (821.25 min)
Velocidade de processamento      : 0.20 samples/s
Loss final do treinamento        : 1.8543

Estatísticas de Memória GPU
============================
Pico de memória reservada        : 9.125 GB
Pico de memória para treinamento : 0.000 GB
Percentual da memória máxima     : 114.120%
Percentual usado no treinamento  : 0.000%
  
AVISO: O uso de memória ultrapassou o limite máximo recomendado!  
  
Treinamento concluido com sucesso!  
  
###############################################################################  
# Preparando o modelo para fazer inferência (previsões)...  
###############################################################################  
  
Modelo preparado para inferência!  
  
###############################################################################  
# Preparando 'Mog's Kittens' para ser tokenizado e passar por inferência...  
###############################################################################  
  
Tokenização realizada com sucesso!  
  
###############################################################################  
# Realizando pergunta para o modelo...  
###############################################################################  
  
Resposta obtida com sucesso!  
  
Resposta do modelo: [The New York Times bestselling author of the beloved Mog 
series returns with a new tale of a cat and her kittens  Mog is a cat who loves 
to sleep in the sun and eat tuna fish But when she has kittens of her own she must 
learn to be a good mother and teach her kittens to be good cats too  This is a 
sweet and funny story about a cat and her kittens that is sure to delight young 
readers  Ages 3 to 7]  
  
###############################################################################  
# Preparando 'Mog's Kittens' para ser tokenizado e passar por inferência...  
###############################################################################  
  
Tokenização realizada com sucesso!  

 The New York Times bestselling author of the beloved Mog series returns with a 
 new story about the little cat who has charmed millions of readers worldwideMog 
 is a cat who likes to be in charge of things Mog likes to be in charge of the 
 other cats Mog likes to be
 in charge of the dog Mog likes to be in charge of the fish Mog likes to be in
  charge of the bird Mog likes to be in charge of the mouse Mog likes to be in 
  charge of the baby Mog likes to be in charge of the worldBut when the baby is
   born Mog finds that being in charge is not as easy as she thought it would beM  
  
Resposta obtida com sucesso!  
  
###############################################################################  
# Salvando o Modelo Treinado em lora_model_llama-3-8b-bnb-4bit...  
###############################################################################  
Unsloth: Merging 4bit and LoRA weights to 16bit...  
Unsloth: Will use up to 0.0 out of 31.73 RAM for saving.  
Unsloth: Saving model... This might take 5 minutes ...  
  0%|                                                    | 0/32 [00:00<?, ?it/s]   
We will save to Disk and not RAM now.  
100%|███████████████████████████████████████████| 32/32 [01:31<00:00,  2.85s/it]  
Unsloth: Saving tokenizer... Done.  
Done.  
  
>>>>>>>>>>>>>>>>>>>>>>>>>>>>> FIM DO FINE-TUNING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  
```

## Testando Modelo Treinado

Nosso código para o treinamento do modelo está disponível no arquivo 
`"ft_test_trained_model.py"`.

O proceso de fine-tuning é feito a partir do arquivo processado 
(`trn_processed.json`) na fase de preparação dos dados para treinamento.

## Exemplo de execução:

```
(.venv) PS C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama> python .\ft_test_trained_model.py  
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.  
🦥 Unsloth Zoo will now patch everything to make training faster!  
  
###############################################################################  
# Criando Configurações do Unsloth  
###############################################################################  
  
Configuração realizada:  
{'attn_implementation': 'flash_attention_2',  
 'device_map': 'auto',  
 'dtype': torch.bfloat16,  
 'load_in_4bit': True,  
 'lora_model': 'lora_model_llama-3-8b-bnb-4bit',  
 'max_seq_length': 8192,  
 'model': 'unsloth/llama-3-8b-bnb-4bit'}  
  
###############################################################################  
# Perguntando ao modelo treinado: ./lora_model_llama-3-8b-bnb-4bit  
###############################################################################  
  
The `load_in_4bit` and `load_in_8bit` arguments are deprecated and will be 
removed in the future versions. Please, pass a `BitsAndBytesConfig` object in 
`quantization_config` argument instead.  
`low_cpu_mem_usage` was None, now default to True since model is quantized.  
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\.venv\Lib\site-packages\accelerate\utils\modeling.py:330: 
UserWarning: expandable_segments not supported on this platform (Triggered internally at 
C:\actions-runner\_work\pytorch\pytorch\pytorch\c10/cuda/CUDAAllocatorConfig.h:28.)
  new_value = value.to(device)  
  
Query:  Girls Ballet Tutu Neon Blue  
  
Resposta do modelo:  
 The perfect gift for the little ballerina in your life this tutu is made of soft 
 nylon tulle and is fully lined with a satin drawstring waistband<|end_of_text|>  
  
Query:  Mog's Kittens  
  
Resposta do modelo:  
 Praise for Mog the Forgetful Catx2018Grandparents are likely to get as much fun 
 out of seeing it again as the new generation of fans just learning to readx2019 
 Choice Magazinex2018A lovely book for all Mogfanciersx2019 The 
 Observerx2018Kerrx2019s watercolours are
 full of humour and expressionx2019 Financial Timesx2018A lovely book for all 
 Mogfanciersx2019 The Observerx2018Kerrx2019s watercolours are full of humour and 
 expressionx2019 Financial Times<|end_of_text|>  
  
Query:  The Prophet  
  
Resposta do modelo:  
 The Prophet is a book of 26 prose poetry essays written in English by 
 LebaneseAmerican artist Kahlil Gibran 1883ndash1931 The Prophet has been 
 translated into over 20 languages and has sold more than 100 million copies 
 worldwide<|end_of_text|>  
  
Query:  The Book of Revelation  
  
Resposta do modelo:  
 The Book of Revelation is the last book of the New Testament and one of the 
 most enigmatic and controversial works in Western literature It is a book of 
 apocalyptic prophecy that predicts the end of the world and the Last Judgment 
 The book is also known as the Apocalypse of John or simply the Apocalypse<|end_of_text|>  
``` 

# 5. Implementação de RAG (Retrieval Augmented Generation)

### **Processo em Duas Etapas:**
1. **Indexação dos dados** no ChromaDB
2. **Consulta do modelo com base indexada**

O processo de RAG (Retrieval Augmented Generation) do modelo envolve duas 
etapas: na primeira, realizamos a indexação dos dados que iremos trabalhar
com o modelo de IA escolhido (`unsloth/llama-3-8b-bnb-4bit`) enquanto na 
segunda, interrogamos o modelo utilizando a base de dados indexada, utilizando RAG.
Uma etapa intermediária foi introduzida, apenas para verificar a indexação dos 
dados de trabalho `(trn_processed.json)`. 

## Indexação dos Dados

Nosso código para a indexação dos dados para realização e RAG está disponível no
arquivo `rag_indexing.py`.

## Fluxo:

  1. Carregamento de trn_processed.json
  2. Quebra em chunks menores
  3. Geração de embeddings
  4. Armazenamento no ChromaDB

O início do processo se dá com o consumo do arquivo `trn_processed.json`. Ele é 
carregado para que seus dados sejam preparados para a indexação na vector store
(`ChromaDB`). Para otimizar as busca, os dados são quebrados em pedaços menores 
(chunks), antes de serem convertidos em vetores de embeddings. Com os embeddings
gerados, chegou o momento de converter os dados processados em documentos que 
serão armazenados na vector store. Tendo os documentos prontamente convertidos, 
chegou o momento de instanciar a vector store e então armazenar documentos e 
embeddings criados.  

## Exemplo de execução:

```
(.venv) PS C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama> python .\rag_indexing.py 

Carregando dados do arquivo processado...
Total de registros gerados:5000
Exemplo de registro: 
{'description': 'High quality 3 layer ballet tutu 12 inches in length',
 'product': 'Girls Ballet Tutu Neon Pink'}

Quebrando documentos JSON em chunks...

Total de chunks gerados: 5000

Exemplo de chunk:
Product: Girls Ballet Tutu Neon Pink - Description: High quality 3 layer ballet tutu 12 inches in length

Criando modelo de embeddings...
Dividindo documentos JSON: 100%|██████████| 5000/5000 [00:00<00:00, 555316.30it/s]
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama\rag_indexing.py:68: 
LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in 
LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists 
in the :class:`~langchain-huggingface package and should be used instead. To use 
it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:
`~langchain_huggingface import HuggingFaceEmbeddings``.
  embeddings_model = HuggingFaceEmbeddings(

Criando modelo de embeddings...

Gerando embeddings para os chunks...

Gerando embeddings...

Total de embeddings gerados: 5000

Exemplo de embedding:
[0.03302772715687752,
 -0.021298706531524658,
 -0.021273870021104813,
 0.052845340222120285,
 -0.019837183877825737,
 -0.03264341503381729,
 0.010283859446644783,
 0.014730839990079403,
 -0.017746785655617714,
 0.003939006011933088,
  6.719978057498912e-35,
 0.037213534116744995,
 -0.10796977579593658,
 0.04069541022181511,
 -0.002336055040359497,
 0.006895443890243769,
 -0.05344033241271973,
 0.08863085508346558,
 -0.03707418590784073,
 0.05612768977880478,
 0.00930757075548172,
 -0.023683171719312668]

Convertendo json_data em lc_documents...

C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama\rag_indexing.py:122: 
LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 
and will be removed in 1.0. An updated version of the class exists in the :class:
`~langchain-chroma package and should be used instead. To use it run `pip install -U :class:
`~langchain-chroma` and import as `from :class:`~langchain_chroma import Chroma``.
  vector_store = Chroma(

Total de lc_documents  gerados: 5000

Exemplo de lc_document:
Document(metadata={'product': 'Girls Ballet Tutu Neon Pink'}, 
page_content='Product: Girls Ballet Tutu Neon Pink - Description: High quality 
3 layer ballet tutu 12 inches in length')

Criando vector_store para o embeddings_model...

vector_store criada!

Populando vector_store...

Gerando UUIDs para cada documento...

Adicionando os documentos no vector_store...

Documentos carregados!
```

## Verificação dos Dados Indexados (Passo Intermediário)
Nosso código para a verificação dos dados que foram indexados anteriormente está
disponível no arquivo `rag_search_vs.py`.

Para realizar a verificação dos dados indexados, utilizamos o modelo de embeddings
criado na indexação dos dados. Em seguida, utilizamos a vector store para realizar
buscas e exibir os resultados.

Os resultados retornados são o 3 mais relevantes ao produto pesquisado. O 
exemplo de execução abaixo mostra os resultados obtidos para o produto
`Girls Ballet Tutu Neon Pink`.  

## Exemplo de execução:

```
(.venv) PS C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama> python .\rag_search_vs.py  
  
Criando modelo de embeddings...  
  
Criando modelo de embeddings...  
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama\rag_indexing.py:68:   
LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in   
LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists   
in the :class:`~langchain-huggingface package and should be used instead. To use it run `  
pip install -U :class:`~langchain-huggingface` and import as   
`from :class:`~langchain_huggingface import HuggingFaceEmbeddings``.  
  embeddings_model = HuggingFaceEmbeddings(  
  
Criando vector_store para o embeddings_model...  
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama\rag_indexing.py:122:   
LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9   
and will be removed in 1.0. An updated version of the class exists in the :class:  
`~langchain-chroma package and should be used instead. To use it run `  
pip install -U :class:`~langchain-chroma` and import as   
`from :class:`~langchain_chroma import Chroma``.  
  vector_store = Chroma(  
  
vector_store criada!  
  
Realizando consultas na vector_store  
  
* Product: Girls Ballet Tutu Neon Blue - Description: Dance tutu for girls ages 
  28 years Perfect for dance practice recitals and performances costumes or just 
  for fun [{'product': 'Girls Ballet Tutu Neon Blue'}]  

* Product: Girls Ballet Tutu Neon Pink - Description: High quality 3 layer ballet 
  tutu 12 inches in length [{'product': 'Girls Ballet Tutu Neon Pink'}]  

* Product: Delphie and the Birthday Show Magic Ballerina Book 6 - Description: 
  x201CDonx2019t be surprised if your child asks for a magical pair of red ballet 
  shoesx201D Telegraph Magazinex201CDelightfulx201D You Magazine Mail on 
  Sundayx201CA delight for any young reader who sees herself as a budding 
  ballerinax201D MumKnowsBestcouk [{'product': 'Delphie and the Birthday Show 
  Magic Ballerina Book 6'}]  
```

## Testando o Modelo com RAG

Nosso código para a realização de consultas ao modelo utilizando RAG está 
disponível no arquivo `rag_model_retriever.py`.

## Exemplo de execução:

```
(.venv) PS C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama> python .\rag_model_retriever.py  
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.  
🦥 Unsloth Zoo will now patch everything to make training faster!  
  
Criando modelo de embeddings...  
  
Criando modelo de embeddings...  
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama\rag_indexing.py:68: 
LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in 
LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists 
in the :class:`~langchain-huggingface package and should be used instead. To use 
it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:
`~langchain_huggingface import HuggingFaceEmbeddings``.  
  embeddings_model = HuggingFaceEmbeddings(  
  
Criando vector_store para o embeddings_model...  
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama\rag_indexing.py:122: 
LangChainDeprecationWarning: The class `Chroma` was deprecated in LangChain 0.2.9 
and will be removed in 1.0. An updated version of the class exists in the :class:
`~langchain-chroma package and should be used instead. To use it run `pip install 
-U :class:`~langchain-chroma` and import as `from :class:`~langchain_chroma import Chroma``.  
  vector_store = Chroma(  
  
vector_store criada!  
  
###############################################################################  
# Criando Configurações do Unsloth  
###############################################################################  
  
Configuração realizada:  
{'attn_implementation': 'flash_attention_2',  
 'device_map': 'auto',  
 'dtype': torch.bfloat16,  
 'load_in_4bit': True,  
 'lora_model': 'lora_model_llama-3-8b-bnb-4bit',  
 'max_seq_length': 8192,  
 'model': 'unsloth/llama-3-8b-bnb-4bit'}  
  
###############################################################################  
# Carregando o modelo e tokenizador para o modelo...  
###############################################################################  
  
==((====))==  Unsloth 2025.1.5: Fast Llama patching. Transformers: 4.49.0.  
   \\   /|    GPU: NVIDIA GeForce RTX 4060 Laptop GPU. Max memory: 7.996 GB. Platform: Windows.  
O^O/ \_/ \    Torch: 2.6.0+cu126. CUDA: 8.9. CUDA Toolkit: 12.6. Triton: 3.2.0  
\        /    Bfloat16 = TRUE. FA [Xformers = 0.0.29.post3. FA2 = False]  
 "-____-"     Free Apache license: http://github.com/unslothai/unsloth   
Unsloth: Fast downloading is enabled - ignore downloading bars which are red colored!  
  
Modelo unsloth/llama-3-8b-bnb-4bit e tokenizador carregados e com sucesso!  
  
###############################################################################  
# Preparando o modelo para fazer inferência (previsões)...  
###############################################################################  
  
Modelo preparado para inferência!  
C:\acmattos\dev\tools\Python\ia4devs\module_03\04_tech_challenge\llama\rag_model_retriever.py:35: 
LangChainDeprecationWarning: The method `BaseRetriever.get_relevant_documents` 
was deprecated in langchain-core 0.1.46 and will be removed in 1.0. Use :meth:
`~invoke` instead.  
  retrieved_docs = retriever.get_relevant_documents(product)  
  
Resposta gerada:  
  
Below is an instruction that describes a task, paired with an input that   
provides further context. Write a response that appropriately completes   
the request.  
  
### Instruction:  
GET THE DESCRIPTION OF THIS PRODUCT: Girls Ballet Tutu Neon Blue  
  
### Input:  
Product: Girls Ballet Tutu Neon Blue - Description: Dance tutu for girls ages 28 
years Perfect for dance practice recitals and performances costumes or just for fun  
Product: Girls Ballet Tutu Neon Pink - Description: High quality 3 layer ballet 
tutu 12 inches in length  
Product: Delphie and the Birthday Show Magic Ballerina Book 6 - Description: 
x201CDonx2019t be surprised if your child asks for a magical pair of red ballet 
shoesx201D Telegraph Magazinex201CDelightfulx201D You Magazine Mail on Sundayx201CA 
delight for any young reader who sees herself as a budding ballerinax201D MumKnowsBestcouk  

### Response:  

The description of this product is: Girls Ballet Tutu Neon Blue - Description: 
Dance tutu for girls ages 28 years Perfect for dance practice recitals and 
performances costumes or just for fun  

Resposta gerada:  
  
Below is an instruction that describes a task, paired with an input that   
provides further context. Write a response that appropriately completes   
the request.  

### Instruction:  
GET THE DESCRIPTION OF THIS PRODUCT: Mog's Kittens  
  
### Input:  
Product: Mogs Kittens - Description: Judith Kerr8217s best8211selling adventures 
of that endearing and exasperating cat Mog have entertained children for more 
than 30 years Now even infants and toddlers can enjoy meeting this loveable 
feline These sturdy little board books8212with their bright simple pictures easy 
text and hand8211friendly formats8212are just the thing to delight the very young 
Ages 6 months82112 years  
Product: Mog and the VET Mog the Cat Books - Description: Praise for Mog the 
Forgetful Catx2018Grandparents are likely to get as much fun out of seeing it 
again as the new generation of fans just learning to readx2019 Choice Magazinex2018A 
lovely book for all Mogfanciersx2019 The ObserverPraise for Goodbye Mogx2018Kerrx2019s 
warmth humour and honesty make this an engaging introduction to a difficult topicx2019 
Financial Timesx2018Believable amusing and movingx2019 Nursery Worldx2018A supremely 
sensitive storyx2019 The Timesx2018The best most consoling book for children on the 
subject or bereavementx2026a joy to readx2019 The Independent on Sunday   
Product: Mog on Fox Night - Description: Praise for Mog the Forgetful 
Catx2018Grandparents are likely to get as much fun out of seeing it again as the 
new generation of fans just learning to readx2019 Choice Magazinex2018A lovely 
book for all Mogfanciersx2019 The ObserverPraise for Goodbye Mogx2018Kerrx2019s 
warmth humour and honesty make this an engaging introduction to a difficult 
topicx2019 Financial Timesx2018Believable amusing and movingx2019 Nursery 
Worldx2018A supremely sensitive storyx2019 The Timesx2018The best most consoling 
book for children on the subject of bereavementx2026a joy to readx2019 The 
Independent on Sunday  
  
### Response:  
  
Mog's Kittens - Description: Judith Kerr8217s best8211selling adventures of that 
endearing and exasperating cat Mog have entertained children for more than 30 
years Now even infants and toddlers  
```

## Comparação das Respostas dos Modelos

| Produto                     | Modelo Original (Foundation Model)                                                                                       | Modelo com Fine-Tuning                                                                                           | Modelo com RAG                                                                                                   |
|-----------------------------|--------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|
| Girls Ballet Tutu Neon Blue | This is a beautiful tutu. It is made of high quality material and it is very comfortable. It is perfect for any occasion and it is very affordable. It is also very durable and it will last for a long time. It is also very easy to maintain and it is very easy to clean. It is also very easy to wear and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off. It is also very easy to put on and it is very easy to take off.                              | The perfect gift for the little ballerina in your life this tutu is made of soft  nylon tulle and is fully lined with a satin drawstring waistband                  | Dance tutu for girls ages 28 years. Perfect for dance practice, recitals and performances, costumes or just for fun. |
| Mog's Kittens               | The product is a pair of socks.                                                                                          | Praise for Mog the Forgetful Catx2018Grandparents are likely to get as much fun out of seeing it again as the new generation of fans just learning to readx2019  Choice Magazinex2018A lovely book for all Mogfanciersx2019 The  Observerx2018Kerrx2019s watercolours are full of humour and expressionx2019 Financial Timesx2018A lovely book for all  Mogfanciersx2019 The Observerx2018Kerrx2019s watercolours are full of humour and  expressionx2019 Financial Times                     | Judith Kerr8217s best8211selling adventures of that endearing and exasperating cat Mog have entertained children for more than 30 years Now even infants and toddlers    |
| The Prophet                 | The Prophet is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money. It is a product that helps you to find the best way to spend your money.                                      | The Prophet is a book of 26 prose poetry essays written in English by  LebaneseAmerican artist Kahlil Gibran 1883ndash1931 The Prophet has been  translated into over 20 languages and has sold more than 100 million copies  worldwide  | In a distant timeless place a mysterious prophet walks the sands At the moment of his departure he wishes to offer the people gifts but possesses nothing The people gather round each asks a question of the heart and the mans wisdom is his gift It is Gibrans gift to us as well for Gibrans prophet is rivaled in his wisdom only by the founders of the worlds great religions On the most basic topicsmarriage children friendship work pleasurehis words have a power and lucidity that in another era would surely have provoked the description divinely inspired Free of dogma free of power structures and metaphysics consider these poetic moving aphorisms a 20thcentury supplement to all sacred traditionsas millions of other readers already haveBrian BruyaThis text refers to theHardcoveredition|
| The Book of Revelation      | The Book of Revelation is a book of the New Testament of the Bible, and its title originated from the first word of the text in the Koine Greek: apokalypsis, meaning "unveiling" or "revelation". The author describes himself as "John" and does not identify himself as the son of Zebedee, the apostle John. The text is a letter to seven churches in the Roman province of Asia, and is a call to the churches to remain faithful to Jesus Christ, and individual letters to each church, with a promise of a swift punishment for Christian communities that are in a state of apostasy. The Book of Revelation is the final book of the New Testament and occupies a central place in Christian eschatology. By tradition, this prophecy was revealed by its author to the apostle John on the island of Patmos, and from its first readers, this prophecy has been accepted as of divine inspiration. The author of Revelation does not identify himself, but introduces his work as "the revelation of Jesus Christ", which he received "by an angel" from God.                                                     | The Book of Revelation is the last book of the New Testament and one of the most enigmatic and controversial works in Western literature It is a book of  apocalyptic prophecy that predicts the end of the world and the Last Judgment  The book is also known as the Apocalypse of John or simply the Apocalypse  | American Baptist pastor Bible teacher and writer Clarence Larkin was born October 28 1850 in Chester Delaware County Pennsylvania He was converted to Christ at the age of 19 and then felt called to the Gospel ministry but the doors of opportunity for study and ministry did not open immediately He then got a job in a bank When he was 21 years old he left the bank and went to college graduating as a mechanical engineer He continued as a professional draftsman for a while then he became a teacher of the blind Later failing health compelled him to give up his teaching career After a prolonged rest he became a manufacturer When he was converted he had become a member of the Episcopal Church but in 1882he became a Baptist and was ordained as a Baptist minister two years later He went directly from business into the ministry His first charge was at Kennett Square Pennsylvania his second pastorate was at Fox Chase Pennsylvania where he remained for 20 years He was not a premillennialist at the time of his ordination but his study of the Scriptures with the help of some books that fell into his hands led him to adopt the premillennialist position He began to make large wall charts which he titled Prophetic Truth for use in the pulpit These led to his being invited to teach in connection with his pastoral work in two Bible institutes During this time he published a number of prophetical charts which were widely circulated When World War I broke out in 1914 he was called on for addresses on The War and Prophecy Then God laid it on his heart to prepare a work on Dispensational Truth or Gods Plan and Purpose in the Ages containing a number of charts with descriptive matter He spent three years of his life designing and drawing the charts and preparing the text The favorable reception it has had since it was first published in 1918 seems to indicate that the world was waiting for such a book Because it had a large and wide circulation in this and other lands the first edition was soon exhausted It was followed by a second edition and then realizing that the book was of permanent value Larkin revised it and expanded it printing it in its present form He went to be with the Lord on January 24 1924This text refers to thePaperbackedition|

---
📌 **Conclusão:**

| Produto                   | Foundation Model | Fine-Tuned Model | RAG Model |
|---------------------------|-----------------|-----------------|-----------|
| **Girls Ballet Tutu** | Genérica e irrelevante | Melhorada, mas imprecisa | Extraída diretamente da base de dados |
| **Mog's Kittens** | Errado | Parcialmente correto | Descrição original extraída da base |


  - O modelo com fine-tuning apresenta respostas mais detalhadas e relevantes em 
    comparação ao modelo original.
  - O modelo com RAG utiliza a base de dados indexada para fornecer respostas mais 
    precisas e contextualizadas, mas está limitado aos dados previamente indexados.
  - O modelo original (Foundation Model) apresenta respostas genéricas e, em alguns 
    casos, completamente irrelevantes, como no exemplo de "Mog's Kittens", onde a 
  - resposta foi "The product is a pair of socks".
  - O modelo com fine-tuning demonstra uma melhoria significativa na qualidade das 
    respostas, mas ainda apresenta redundâncias ou informações desnecessárias em 
    alguns casos.
  - O modelo com RAG é altamente dependente da qualidade e abrangência dos dados 
    indexados. Ele é ideal para cenários onde as informações relevantes estão 
    contidas em uma base de dados específica.