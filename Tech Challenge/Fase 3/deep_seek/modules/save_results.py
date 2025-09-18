import json

def save_results(json_results, filename="amazon_titles_reasoning.json"):
    """This function saves the results to a JSON file."""
    with open(filename, "w", encoding="utf-8") as result_file:
        print("======================= Products saved successfully in JSON format =======================")
        result_file.write(json.dumps(json_results, ensure_ascii=False, indent=4))

    print("======================= Completed =======================")
