from data_collection.models import Game
from data_collection.utils import get_qa_history_prompt, call_gpt4, get_secret_message


def get_question_candidates_generation_prompt(
    game: Game,
    candidates_to_generate: int = 10,
    include_answer: bool = False
):
    if include_answer:
        secret_message = f"\n{get_secret_message(game)}\n"
    else:
        secret_message = ""
    return f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(game)}
    
You need to generate {candidates_to_generate} candidates for the next question to ask.
You should mix safe questions with more creative and non-orthodox ones.
Safe questions are usually ones that split the space of possibilities in half.
If you have a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.
Consider asking non-standard questions like "Does the keyword consist of a single word?, however only do it if you don't know it yet, and if you think that it's important to know at this point in the game.
Consider asking questions like "Does the keyword start with letter 'A'?" or "Does the keyword contain letter 'E'?". However, don't overdo it, focus on more substantial questions.
{secret_message}
Format your answer like this: output every question on a new line. Don't output anything else.
"""


async def get_question_candidates(
    client,
    game,
    include_answer: bool = False
):
    response = await call_gpt4(
        client,
        get_question_candidates_generation_prompt(game, include_answer=include_answer),
        game_state=game,
        request_type="questions_candidates" + ("_with_answer" if include_answer else "")
    )
    questions = response.strip().split("\n")
    return questions
