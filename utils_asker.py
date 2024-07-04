from utils import generate_answer


def ask(
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
So focus your area of search on these options. and give smart questions that narrows down the search space\
"""

    ask_prompt = sys_prompt + """Your role is to find the word by asking him up to 20 questions, your questions to be valid must have only a 'yes' or 'no' answer.
To help you, here's an example of how it should work assuming that the keyword is Morocco:
Example:
<you: is it a place?
user: yes
you: is it in europe?
user: no
you: is it in africa?
user: yes
you: do most people living there have dark skin?
user: no
user: is it a country name starting by m ?
you: yes
you: is it Morocco?
user: yes.>

The user has chosen the word, ask your question!
please be concise and not verbose, give only one question, nothing else!
"""
    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{ask_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if len(obs.questions) >= 1:
        for q, a in zip(obs.questions, obs.answers):
            chat_template += f"{q}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            chat_template += f"{a}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    output = generate_answer(chat_template, model, tokenizer, id_eot)
    return output
