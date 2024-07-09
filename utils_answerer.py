from utils import generate_answer, SYSTEM_PROMPT_ANSWERER, get_qa_history_prompt


def answer(
    obs,
    model,
    tokenizer,
    id_eot,
):
    sys_prompt = SYSTEM_PROMPT_ANSWERER
    user_prompt = f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(obs, include_guesses=True)}

The guessed keyword is "{obs.keyword}".

Another player is asking: "{obs.questions[-1]}".

You goal is to answer this question about the keyword as truthfully as possible.
You can only answer YES or NO. Nothing else. Even if you don't know, even if there's no simple yes/no answer - you still HAVE TO answer YES or NO.
Given this harsh constraint, still do your best to be useful, truthful, and helpful to the other player.

Format your answer like this: First, reason about your answer for 1-2 sentences. If the answer is obvious - state it. If not - reason about whether YES or NO would be more helpful and truthful.
Never reason for more than 4 sentences! After 4 sentences just get to the answer (YES or NO) on a new line!
Then, your last line should be the answer, only YES or NO.
Don't output anything else.
"""

    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>user<|end_header_id|>\n\n"
    chat_template += f"{user_prompt}<|eot_id|>"

    output = generate_answer(chat_template, model, tokenizer, id_eot)
    lines = output.strip().split("\n")
    reasoning = " ".join(lines[:-1])
    answer = lines[-1].lower().strip()
    answer = "".join([c for c in answer if c.isalpha()])
    try:
        assert answer in ["yes", "no"], f"Invalid answer: {answer}"
    except:
        return "no"
    return answer
