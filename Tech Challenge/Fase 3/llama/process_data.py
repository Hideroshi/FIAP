import json
import re
from tqdm import tqdm

def count_records(file):
   total_lines = sum(1 for line in open(file, 'r', encoding='utf-8'))
   return total_lines

def read_json_file(file, total_lines):
   json_data = []
   with open(file, 'r', encoding='utf-8') as file:
      for line in tqdm(file, total=total_lines, desc="Lendo os registros do arquivo              "):
         json_datum = json.loads(line)
         json_data.append(json_datum)   
   return json_data

# Função para remover caracteres especiais
def clean_text(text):
    # Remover caracteres especiais (mantém apenas letras, números e espaços)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

def remove_invalid_records(json_data):
   filtered_records = []
   # Processa cada registro
   for item in tqdm(json_data, desc="Processando registros lidos                "):
      title = item["title"] 
      content = item["content"] 
      # Apenas adiciona se ambos os campos não estiverem vazios
      if title and content and len(content) >= 100:
         filtered_records.append(process_item(title, content))
   return filtered_records

def process_item(product, description):
   return {
        'product': clean_text(product),
        'description': clean_text(description)
   }

def write_json_file(filtered_records, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('[\n')
        for i, item in enumerate(tqdm(filtered_records, desc="Salvando registros processado em arquivo   ")):
            file.write(' ')
            json.dump(item, file, ensure_ascii=False)
            if i < len(filtered_records) - 1:
                file.write(',\n')
            else:
                file.write('\n')      
        file.write(']')  

def process_json_file(input_file, output_file):
   print() 
   # Conta linhas do arquivo
   total_lines = count_records(input_file)

   # Le o arquivo JSON de entrada
   json_data = read_json_file(input_file, total_lines)   

   # Remove registros invalidos
   filtered_records = remove_invalid_records(json_data)
    
   # Escreve os registros filtrados em um novo arquivo JSON
   write_json_file(filtered_records, output_file) 

   print(f"Total de registros no arquivo original     : {total_lines:,}")
   print(f"Total de registros processados (não vazios): {len(filtered_records):,}")
   print(f"Arquivo processado salvo em                : {output_file}")
   print() 

# Executando a função principal
if __name__ == "__main__":
    process_json_file('./trn.json', './trn_processed.json')

