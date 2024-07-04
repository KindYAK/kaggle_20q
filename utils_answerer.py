from utils import generate_answer


def answer(
    obs,
    model,
    tokenizer,
    id_eot,
):
    sys_prompt = f"""
You are a helpful AI assistant, and your are very smart in playing 20 questions game,
the role of the user is to guess the word by asking you up to 20 questions, your answers to be valid must be a 'yes' or 'no', any other answer is invalid and you lose the game.
Know that the user will always guess a word belonging to one of the following 3 categories:
1. a place
2. a person
3. a thing
so make sure you understand the user's question and you understand the keyword you're playig on.
for now the word that the user should guess is: "{obs.keyword}", it is of category "{obs.category}",
to help you, here's an example of how it should work assuming that the keyword is Morocco in the category "place":
examle:
<user: is it a place?
you: yes
user: is it in europe?
you: no
user: is it in africa?
you: yes
user: do most people living there have dark skin?
you: no
user: is it a country name starting by m ?
you: yes
user: is it Morocco?
you: yes.>
"""

    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>user<|end_header_id|>\n\n"
    chat_template += f"{obs.questions[0]}<|eot_id|>"
    chat_template += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if len(obs.answers) >= 1:
        for q, a in zip(obs.questions[1:], obs.answers):
            chat_template += f"{a}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
            chat_template += f"{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    output = generate_answer(chat_template, model, tokenizer, id_eot)
    return output
