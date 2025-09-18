from modules.process_product import process_product

def handle_process_inputs(amazon_products):
    """This function handles the products and returns a list of results."""
    list_of_processed_questions_and_answers = []
    for llm_data in amazon_products:
        try:
            result = process_product(llm_data)
            if result:
                list_of_processed_questions_and_answers.append(result)
        except (ValueError, AttributeError) as error:
            print(f"Error processing product {llm_data['title'][:30]}...: {error}")
    return list_of_processed_questions_and_answers
