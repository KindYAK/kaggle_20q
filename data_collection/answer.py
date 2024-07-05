from data_collection.models import Game
from data_collection.utils import get_qa_history_prompt, call_gpt4


def get_question_answer_prompt(
    game: Game,
    keyword: str,
    question: str,
):
    return f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(game, include_guesses=True)}

The guessed keyword is "{keyword}".

Another player is asking: "{question}".

You goal is to answer this question about the keyword as truthfully as possible.
You can only answer YES or NO. Nothing else. Even if you don't know, even if there's no simple yes/no answer - you still HAVE TO answer YES or NO.
Given this harsh constraint, still do your best to be useful, truthful, and helpful to the other player.

Format your answer like this: First, reason about your answer for 1-3 sentences. If the answer is obvious - state it. If not - reason about whether YES or NO would be more helpful and truthful.
Then, your last line should be the answer, only YES or NO.
Don't output anything else.
"""


async def answer(client, game: Game, keyword: str, question: str) -> tuple[str, str]:
    response = await call_gpt4(client, get_question_answer_prompt(game, keyword, question), game_state=game, request_type="answer")
    questions = response.strip().split("\n")
    reasoning = " ".join(questions[:-1])
    answer = questions[-1].lower().strip()
    answer = "".join([c for c in answer if c.isalpha()])
    try:
        assert answer in ["yes", "no"], f"Invalid answer: {answer}"
    except:
        return "Error occured, returning default answer", "no"
    return reasoning, answer
