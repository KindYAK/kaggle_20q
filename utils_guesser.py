from utils import generate_answer


def guess(
    obs,
    model,
    tokenizer,
    id_eot,
):
    sys_prompt = """
You are a helpful AI assistant, and your are very smart in playing 20 questions game,
the user is going to think of a word, it can be only one of the following 3 categories:
1. a place
2. a person
3. a thing
So focus your area of search on these options. and give smart questions that narrows down the search space\n
"""

    conv = ""
    for q, a in zip(obs.questions, obs.answers):
        conv += f"""Question: {q}\nAnswer: {a}\n"""
    guess_prompt = sys_prompt + f"""
So far, the current state of the game is as follows:\n{conv}
based on the conversation, can you guess the word, please give only the word, no verbosity around
"""
    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{guess_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    output = generate_answer(chat_template, model, tokenizer, id_eot)
    return output
