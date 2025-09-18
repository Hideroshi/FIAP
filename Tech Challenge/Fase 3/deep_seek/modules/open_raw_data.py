import json
import os

def open_raw_data(file_path, max_number_of_products=2000):
    """This function opens the raw data file and returns a list of products."""
    amazon_products = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    absolute_path = os.path.join(root_dir, file_path)

    with open(absolute_path, 'r', encoding='utf-8') as amazon_products_file:
        for i, line in enumerate(amazon_products_file):
            if i >= max_number_of_products:
                break
            json_data = json.loads(line)
            if json_data.get("title") and json_data.get("content"):
                #only assign the product if product has a title and description
                clean_title = ' '.join(json_data["title"].split()).strip()
                clean_content = ' '.join(json_data["content"].split()).strip()

                amazon_products.append({
                    'title': clean_title,
                    'context': clean_content
                })

        return amazon_products
