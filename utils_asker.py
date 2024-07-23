from utils import generate_answer, SYSTEM_PROMPT_ASKER, get_qa_history_prompt, generate_answers_batch


def create_ask_prompt(obs):
    sys_prompt = SYSTEM_PROMPT_ASKER
    user_prompt = f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(obs)}
    
Your goal is to choose the best next question.
Safe questions are usually ones that split the space of possibilities in half given the previous answers.
If you have a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.

You should keep balance of safe questions and more creative and non-orthodox ones.
Safe questions are usually ones that split the space of possibilities in half.
If you have a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.
Consider asking non-standard questions like "Does the keyword consist of a single word?, however only do it if you don't know it yet, and if you think that it's important to know at this point in the game.
Consider asking questions like "Does the keyword start with letter 'A'?" or "Does the keyword contain letter 'E'?". However, don't overdo it, focus on more substantial questions.

In any case, try to make reason about what decision is the best here, balance risk, and choose the best next question. Act as a competetive extremely smart player that you are.

Don't forget that some inconsistencies in answers can be caused by errors in the answering LLM.

Format your answer like this: Your last and only line should be the question that you want to ask. Just output the question, not "Question: ..." or "I pick question..."
Don't output anything else!
"""
    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>user<|end_header_id|>\n\n"
    chat_template += f"{user_prompt}<|eot_id|>"
    return chat_template


def call_model(chat_template, model, tokenizer, id_eot):
    return generate_answer(chat_template, model, tokenizer, id_eot, system_prompt=SYSTEM_PROMPT_ASKER)


def post_process(output):
    lines = output.strip().split("\n")
    reasoning = " ".join(lines[:-1])
    question = lines[-1]
    return question


def ask(obs, model, tokenizer, id_eot):
    chat_template = create_ask_prompt(obs)
    output = call_model(chat_template, model, tokenizer, id_eot)
    return post_process(output)


def ask_batch(obs_list, model, tokenizer, id_eot):
    templates = [create_ask_prompt(obs) if obs is not None else None for obs in obs_list]
    outputs = generate_answers_batch(templates, tokenizer, model, id_eot)
    return [post_process(output) if output is not None else None for output in outputs]
