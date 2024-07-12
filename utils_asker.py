from utils import generate_answer, SYSTEM_PROMPT_ASKER, get_qa_history_prompt


def generate_candidates(obs, model, tokenizer, id_eot, candidates_to_generate=5, max_new_tokens_reason: int = 500):
    sys_prompt = SYSTEM_PROMPT_ASKER
    user_prompt = f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(obs)}

You need to generate {candidates_to_generate} best candidates for the next question to ask.
You should mix safe questions with more creative and non-orthodox ones.
Safe questions are usually ones that split the space of possibilities in half.
If you have a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.
Consider asking non-standard questions like "Does the keyword consist of a single word?, however only do it if you don't know it yet, and if you think that it's important to know at this point in the game.
Consider asking questions like "Does the keyword start with letter 'A'?" or "Does the keyword contain letter 'E'?". However, don't overdo it, focus on more substantial questions.

Format your answer like this: output every question on a new line.
Then, reason for 2-3 sentences about which question is the best one, given the question-answer history.
"""

    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>user<|end_header_id|>\n\n"
    chat_template += f"{user_prompt}<|eot_id|>"
    output = generate_answer(chat_template, model, tokenizer, id_eot, max_new_tokens=max_new_tokens_reason, system_prompt=sys_prompt)
    return output.strip().split("\n")


def ask(
    obs,
    model,
    tokenizer,
    id_eot,
    max_new_tokens_reason: int = 500,
):
    candidates = generate_candidates(obs, model, tokenizer, id_eot, max_new_tokens_reason=max_new_tokens_reason)
    candidates_message = "\n".join(candidates)
    # print("!", candidates_message) # TODO DEBUG
    sys_prompt = SYSTEM_PROMPT_ASKER
    user_prompt = f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(obs)}
    
Here are some candidates for the next question and some considerations for choosing the best one:
{candidates_message}

You goal is to pick the best next question.
Safe questions are usually ones that split the space of possibilities in half given the previous answers.
If you have a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.

In any case, try to make reason about what decision is the best here, balance risk, and choose the best next question. Act as a competetive extremely smart player that you are.

Don't forget that some inconsistencies in answers can be caused by errors in the answering LLM.

Format your answer like this: Your last and only line should be the questions that you've picked. Just output the question, not "Question: ..." or "I pick question..."
Don't output anything else.
"""

    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>user<|end_header_id|>\n\n"
    chat_template += f"{user_prompt}<|eot_id|>"
    output = generate_answer(chat_template, model, tokenizer, id_eot, system_prompt=sys_prompt)
    lines = output.strip().split("\n")
    reasoning = " ".join(lines[:-1])
    question = lines[-1]
    return question
