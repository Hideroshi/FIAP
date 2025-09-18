# FIAP Pós-Tech – Tech Challenge

Este repositório reúne os projetos desenvolvidos nas **5 Fases** do *Tech Challenge* da Pós-Tech FIAP.  
Cada fase aborda um tema e desafio diferente, utilizando tecnologias variadas.

---

## 📚 Fases do Tech Challenge

### 🔹 [Fase 1](Tech%20Challenge/Fase%201)
**Resumo:** *O objetivo é o desenvolvimento de um modelo preditivo de regressão para estimar os custos de planos de saúde individuais. O foco foi realizar previsões utilizando um modelo previamente treinado. A tarefa envolve carregar dados, aplicar o modelo para gerar previsões, e analisar os resultados. É uma etapa de prototipação: testar se o modelo funciona bem com dados de entrada novos, visualizar seus erros e acertos, e possivelmente exportar resultados para relatórios ou para avaliação futura*  
**Tecnologias principais:** * Python, Jupyter Notebook, Pandas, NumPy, scikit-learn, Matplotlib/Seaborn *

---

### 🔹 [Fase 2](Tech%20Challenge/Fase%202)
**Resumo:** * O projeto aborda o problema de alocação de ordens de serviço (OS) de manutenção para operadores de forma otimizada. Utilizando um algoritmo genético, o sistema realiza o pareamento entre as competências dos operadores e as exigências das ordens, visando maximizar a produtividade, reduzir horas extras e garantir o cumprimento de prazos. Além do motor de otimização, foi criada uma interface interativa com Tkinter e TTKBootstrap, incluindo relatórios em tempo real e exportação dos resultados em CSV/Excel. *  
**Tecnologias principais:** * Python, Algoritmos Genéticos, Tkinter e TTKBootstrap (interface gráfica), PyGame (visualização interativa das alocações), Pandas e NumPy *

---

### 🔹 [Fase 3](Tech%20Challenge/Fase%203)
**Resumo:** * O objetivo desta fase foi explorar fine-tuning de modelos de linguagem (LLMs) para criar um assistente virtual de compras capaz de responder perguntas e fornecer recomendações sobre produtos da Amazon.

Foram conduzidos dois experimentos:

Llama: usado como base para avaliação inicial de adaptação de LLMs ao domínio de e-commerce.

DeepSeek: aplicado com um processo de fine-tuning supervisionado usando LoRA para otimizar recursos. O dataset de amostragem foi extraído do AmazonTitles-1.3MM, tratado e transformado em pares de pergunta, raciocínio complexo (CoT) e resposta, possibilitando um ajuste específico ao contexto de compras online.

O projeto utilizou ferramentas de monitoramento (Weights & Biases), pipelines de pré-processamento em Python e disponibilização dos modelos/datasets no Hugging Face. O resultado foi um modelo especializado que apresenta raciocínios mais assertivos e respostas contextualizadas sobre produtos, validando a eficácia do processo de fine-tuning. *  
**Tecnologias principais:** * Python, PyTorch, Transformers (Hugging Face), Hugging Face Datasets & Models, DeepSeek (unsloth/DeepSeek-R1-Distill-Llama-8B), Ollama (execução local de LLMs), LoRA (Low-Rank Adaptation), Weights & Biases (monitoramento de métricas) *

---

### 🔹 [Fase 4](Tech%20Challenge/Fase%204)
**Resumo:** *Projeto de análise de vídeo com IA que realiza reconhecimento facial, detecção de emoções, análise de movimentos corporais e transcrição de áudio. O sistema processa vídeos frame a frame para identificar pessoas, expressões emocionais e atividades, gerando relatórios detalhados com estatísticas de aparições, distribuição de emoções e detecção de anomalias *  
**Tecnologias principais:** * OpenCV, DeepFace, MediaPipe, Dlib, TensorFlow, Vosk, Pandas, NumPy *

---

### 🔹 [Fase 5](Tech%20Challenge/Fase%205)
**Resumo:** * Desenvolvimento de um sistema de detecção de componentes em diagramas de arquitetura AWS utilizando modelos YOLO11 (Nano, Small, Medium) treinados com dataset aprimorado, seguido de geração automatizada de relatórios de ameaças STRIDE e uma aplicação demo com Streamlit e RAG (Retrieval-Augmented Generation) para análise contextualizada de segurança. *  
**Tecnologias principais:** * YOLO11, Ultralytics, PyTorch, OpenCV, Streamlit, FAISS, Ollama, Flask, Telegram API, N8N, Digital Ocean *

---

## 🚀 Como usar este repositório

1. Clone o repositório:
   ```bash
   git clone https://github.com/Hideroshi/FIAP.git

2. Acesse a fase desejada:
    ```bash
    cd FIAP/Tech\ Challenge/Fase\ 3

2. Acesse a fase desejada:
    Siga as instruções no README.md de cada fase caso exista, mas todas as fases possuem arquivo de descrição do desafio e arquivo de solução.

👨‍💻 Integrantes:
    # Aurelio Thomasi Jr
    # André Mattos
    # Leonardo Ramires
    # Lucas Arruda
    # Pedro Marins


📄 Licença:
    Este repositório está sob a licença GPL-3.0.

