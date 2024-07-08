from data_collection.models import Game
from data_collection.utils import get_qa_history_prompt, call_gpt4


def get_question_picker_prompt(
    game: Game,
    candidates: list[str],
):
    candidates_message = "\n".join(candidates)
    return f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(game)}
    
Here are some candidates for the next question:
{candidates_message}

You goal is to pick the best next question.
Safe questions are usually ones that split the space of possibilities in half given the previous answers.
If you have a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.

In any case, try to make reason about what decision is the best here, balance risk, and choose the best next question. Act as a competetive extremely smart player that you are.

Don't forget that some inconsistencies in answers can be caused by errors in the answering LLM.

Format your answer like this: First, reason about your decision for 2-4 sentences. Summarize what you know, summarize space of possibilities, reason about what question would be the most beneficial.
Then, your last line should be the questions that you've picked.
Don't output anything else.
"""


async def pick_question(client, game: Game, candidates: list[str]) -> tuple[str, str]:
    response = await call_gpt4(client, get_question_picker_prompt(game, candidates), game_state=game, request_type="question_pick")
    lines = response.strip().split("\n")
    reasoning = " ".join(lines[:-1])
    question = lines[-1]
    return reasoning, question
