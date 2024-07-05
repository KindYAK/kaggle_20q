from data_collection.models import Game
from data_collection.utils import get_qa_history_prompt, call_gpt4


def get_question_scoring_prompt(
    game: Game,
    candidates: list[str],
):
    candidates_message = "\n".join(candidates)
    return f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(game)}
    
Here are some candidates for the next question:
{candidates_message}
    
For every question you need to assess it's informativeness.
Here's the example:
If we already now that answer to "Is it a fruit?" is "yes", we would assess informativeness for candidate questions as follows:
Is it sweet? (5% - 95%)
Is it red? (70% - 30%)
The reason for that is that if we know that it is fruit, most fruits are sweet (95%-5%).
And we can assess that ~30% of fruits are red, the rest are not.

You need to take existing question-answer history into account!

Format your answer like this: output every question on a new line. Output score as presented in example above in parentheses.
Don't output anything else.
"""


async def get_question_candidates_scored(client, game: Game, candidates: list[str]):
    response = await call_gpt4(client, get_question_scoring_prompt(game, candidates), game_state=game, request_type="question_candidates_entropy")
    questions = response.strip().split("\n")
    return questions
