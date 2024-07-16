from utils import generate_answer, generate_answers_batch, SYSTEM_PROMPT_ANSWERER, get_qa_history_prompt


def create_prompt(obs):
    sys_prompt = SYSTEM_PROMPT_ANSWERER
    user_prompt = f"""You are playing "20 Questions" game.

{get_qa_history_prompt(obs, include_guesses=True)}

The guessed keyword is "{obs['keyword']}".

Another player is asking: "{obs['questions'][-1]}".

Your goal is to answer this question about the keyword as truthfully as possible.
You can only answer YES or NO. Nothing else. Even if you don't know, even if there's no simple yes/no answer - you still HAVE TO answer YES or NO.
Given this harsh constraint, still do your best to be useful, truthful, and helpful to the other player.

Format your answer like this: Your last and only line should be the answer, only YES or NO.
Don't output anything else.
"""
    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>user<|end_header_id|>\n\n"
    chat_template += f"{user_prompt}<|eot_id|>"
    print("!!! prompt")
    print(chat_template)
    # TODO
    return chat_template


def call_model(chat_template, model, tokenizer, id_eot):
    return generate_answer(chat_template, model, tokenizer, id_eot, system_prompt=SYSTEM_PROMPT_ANSWERER)


def post_process(output):
    lines = output.strip().split("\n")
    reasoning = " ".join(lines[:-1])
    answer = lines[-1].lower().strip()
    answer = "".join([c for c in answer if c.isalpha()])
    try:
        assert answer in ["yes", "no"], f"Invalid answer: {answer}"
    except AssertionError as e:
        print("!!! ERROR", answer) # DEBUG TODO
        print(output)
        return "no"
    return answer


def answer(obs, model, tokenizer, id_eot):
    chat_template = create_prompt(obs)
    output = call_model(chat_template, model, tokenizer, id_eot)
    return post_process(output)


def answer_batch(obs_list, model, tokenizer, id_eot):
    templates = [create_prompt(obs) if obs is not None else None for obs in obs_list]
    outputs = generate_answers_batch(templates, tokenizer, model, id_eot)
    return [post_process(output) for output in outputs]
