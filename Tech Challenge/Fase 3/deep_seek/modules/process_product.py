import json

from modules.deep_seek import generate_data_for_fine_tuning
def process_product(product):
    """This function calls the LLM Model to generate the list of questions and answers for the product."""
    response = generate_data_for_fine_tuning(product)
    try:
        # Extract JSON from the string
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        json_str = response[json_start:json_end]

        # Parse JSON
        extracted_json = json.loads(json_str)
        print("==================================JSON GENERATED ==================================")
        print(extracted_json)
        return extracted_json
    except (ValueError, AttributeError) as error:
        print(f"Error parsing JSON for product: {error}")
        return None
