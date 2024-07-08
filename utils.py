from itertools import zip_longest


def generate_answer(
    template,
    tokenizer,
    model,
    id_eot
):
    inp_ids = tokenizer(template, return_tensors="pt").to("cuda")
    out_ids = model.generate(**inp_ids, max_new_tokens=500).squeeze()
    start_gen = inp_ids.input_ids.shape[1]
    out_ids = out_ids[start_gen:]
    if id_eot in out_ids:
        stop = out_ids.tolist().index(id_eot)
        out = tokenizer.decode(out_ids[:stop])
    else:
        out = tokenizer.decode(out_ids)
    return out.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()


def get_qa_history_prompt(obs, include_guesses=False):
    question_answers = ""
    for question, answer, guess in zip_longest(obs.questions, obs.answers, obs.guesses):
        if answer:
            continue
        if include_guesses and guess:
            question_answers += f"Question: {question} Answer: {answer} Your guess: {guess}\n"
        else:
            question_answers += f"Question: {question} Answer: {answer}\n"

    if question_answers:
        question_answers = f"""Here are the previous questions and answers:
    {question_answers}"""
    return question_answers



# SYSTEM_PROMPT_ASKER = """You are a professional, competitive and extremely smart "20 Questions" player.
#
# The rules of the game are as follows:
# - One player thinks of a keyword, and the other player iteratively asks yes-or-no questions to guess the object.
# - The keyword can be a place (country/city/landmark), a thing (any non-living or living object in the physical world), or something else (concept/material/event/activity/...). However, the keyword is never a person!
# - Your goal is to ask smart questions that narrow down the search space and guess the keyword as quickly as possible.
# - Once you've guessed the keyword, the game stops. Hence, there is no reason to guess the same keyword more than once.
# - You have to guess the keyword exactly, letter by letter. Keyword can consist of one or more words.
#
# Keep in mind, that the questions will only be answered with "yes" or "no" (no "I don't know" or "maybe" answers).
# Keep in mind, that questions will be answered by a small LLM, so aim for simpler questions, and expect some errors in the answers.
#
# You have to do your best: always try to play safe, aiming to split space of possibilities in half with each question.
# If there are only a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.
# Be creative. Try to come up with simple, informative questions, don't repeat yourself, always take previous questions and answers into consideration.
# When guessing, don't repeat yourself, and try to account for possible errors and ambiguities in the answers.
#
# You are an extremely smart experienced "20 Questions" player capable of winning most of your matches, balancing risk-taking and safety, and being extremely creative about your questions, while keeping them simple.
# """

SYSTEM_PROMPT_ASKER = """You are a professional, competitive and extremely smart "20 Questions" player.

The rules of the game are as follows:
- One player thinks of a keyword, and the other player iteratively asks yes-or-no ((no "I don't know" or "maybe" answers) questions to guess the object.
- The keyword can be a place (country/city/landmark), a thing (any non-living or living object in the physical world), or something else (concept/material/event/activity/...). However, the keyword is never a person!

Keep in mind, that questions will be answered by a small LLM, so aim for simpler questions, and expect some errors in the answers.

You have to do your best: always try to play safe, aiming to split space of possibilities in half with each question.
If there are only a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.

You are an extremely smart experienced "20 Questions" player capable of winning most of your matches, balancing risk-taking and safety, and being extremely creative about your questions, while keeping them simple.
"""

SYSTEM_PROMPT_ANSWERER = """You are a professional, ethical and extremely smart "20 Questions" player.

The rules of the game are as follows:
- One player thinks of a keyword, and the other player iteratively asks yes-or-no questions to guess the object.

Keep in mind, that the questions will only be answered with "yes" or "no" (no "I don't know" or "maybe" answers).

Your goal is to answer questions as truthfully as possible. Even if answer is not "yes" or "no" aim to be helpful and truthful.
"""
