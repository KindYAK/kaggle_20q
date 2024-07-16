from utils import generate_answer, SYSTEM_PROMPT_ASKER, get_qa_history_prompt


def guess(
    obs,
    model,
    tokenizer,
    id_eot,
):
    sys_prompt = SYSTEM_PROMPT_ASKER

    user_prompt = f"""You are playing "20 Questions" game.

{get_qa_history_prompt(obs, include_guesses=True)}

Your goal is to make a guess about the keyword given the history of questions and answers.
History above contains previous guesses. When the guess is correct, the game stops. So, there's not reason to try a guess that was already made!
Also, keep in mind that guess only counts if it's correct letter-by-letter. So, if you have strong reasons to believe that one of the previous guesses is correct, but was
phrased differently - you can try to guess it again. 
Don't forget that some inconsistencies in answers can be caused by errors in the answering LLM.
The keyword can consist of one or more words.

Don't forget, that you should never guess whatever was already guessed before! You should always try new options to gain more information!
Only output the same guess if there might be another accepted phrasing of the same keyword (which is usually the case when there are several words).

Format your answer like this: only output the keyword that you want to guess, nothing else. Don't say anything like "My guess is {{keyword}}" - always just "{{keyword}}"
"""

    chat_template = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_prompt}<|eot_id|>"""
    chat_template += "<|start_header_id|>user<|end_header_id|>\n\n"
    chat_template += f"{user_prompt}<|eot_id|>"

    output = generate_answer(chat_template, model, tokenizer, id_eot, system_prompt=sys_prompt)
    print("!!!", output) # TODO DEBUG
    lines = output.strip().split("\n")
    reasoning = " ".join(lines[:-1])
    guess = lines[-1].lower().strip()
    return "".join([c for c in guess if c.isalnum() or c.isspace() or c == "-"])
