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

Each candidate was scored in format (N% - (100-N)%), indicating ratio in which space of possibilities is split if the questions is asked.
Here's the example:
If we already now that answer to "Is it a fruit?" is "yes", we would assess informativeness for candidate questions as follows:
Is it sweet? (5% - 95%)
Is it red? (70% - 30%)
The reason for that is that if we know that it is fruit, most fruits are sweet (95%-5%).
And we can assess that ~30% of fruits are red, the rest are not.

You goal is to pick the best next question.
Safe questions are usually ones that split the space of possibilities in half (~50% - 50%)
If you have a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.

In any case, try to make reason about what decision is the best here, balance risk, and choose the best next question. Act as competetive extremely smart player that you are.

Don't forget that some inconsistencies in answers can be caused by errors in the answering LLM.

Format your answer like this: First, reason about your decision for 2-4 sentences. Summarize what you know, summarize space of possibilities, reason about what question would be the most beneficial.
Then, your last line should be the questions that you've picked (without the "(N% - (100-N)%)").
Don't output anything else.
"""


async def pick_question(client, game: Game, candidates: list[str]) -> tuple[str, str]:
    response = await call_gpt4(client, get_question_picker_prompt(game, candidates), game_state=game, request_type="question_pick")
    questions = response.strip().split("\n")
    reasoning = " ".join(questions[:-1])
    question = questions[-1]
    return reasoning, question
