from ollama import chat
from ollama import ChatResponse

def generate_data_for_fine_tuning(json_data: dict):
    """This function generates a JSON object including the context of a chat between a user and an assistant.
    The JSON object includes the question, reasoning, and answer.
    The JSON object is used for fine-tuning DeepSeek model.
    """
    prompt_for_question = f"""
    You are a helpful assistant that generates data for fine-tuning DeepSeek model.
    You will be given a product title and context (description).
    You need to generate a question, reasoning, and answer based on the product title and context.
    The format should be as follows:
    Question: <question>
    Complex_CoT: <reasoning>
    Response: <answer>
    Your response should be a JSON object in the format above, assuming that an user is looking for a product on Amazon.
    Your response to this request should only include the JSON result, no description should be added.
    The question should be a question that an user would ask to find the product on Amazon.
    The reasoning should be a detailed reasoning for the question, with all the steps and the final answer.
    The answer should be the answer to the question.

    Here is the product title and context:
    Title: {json_data['title']}
    Context: {json_data['context']}
    """

    response: ChatResponse = chat(model='deepseek-r1:14b', messages=[
        {
            'role': 'user',
            'content': prompt_for_question,
        },
    ])
    return response['message']['content']
