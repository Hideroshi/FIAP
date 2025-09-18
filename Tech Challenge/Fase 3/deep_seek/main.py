"""Main function of the script to generate the list of questions and answers for a product."""

from modules.handle_process_inputs import handle_process_inputs
from modules.save_results import save_results
from modules.open_raw_data import open_raw_data

if __name__ == "__main__":
    raw_data = open_raw_data("trn.json")
    print(raw_data)
    processed_data = handle_process_inputs(raw_data)
    save_results(processed_data)
