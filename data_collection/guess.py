from data_collection.models import Game
from data_collection.utils import get_qa_history_prompt, call_gpt4, get_secret_message


def get_question_guess_prompt(
    game: Game,
    include_answer: bool = False
):
    if include_answer:
        secret_message = f"\n{get_secret_message(game)}\n"
    else:
        secret_message = ""
    return f"""You are playing "20 Questions" game.
    
{get_qa_history_prompt(game, include_guesses=True)}

You goal is to make a guess about the keyword given the history of questions and answers.
History above contains previous guesses. When the guess is correct, the game stops. So, there's not reason to try a guess that was already made!
Also, keep in mind that guess only counts if it's correct letter-by-letter. So, if you have strong reasons to believe that one of the previous guesses is correct, but was
phrased differently - you can try to guess it again. 
Don't forget that some inconsistencies in answers can be caused by errors in the answering LLM.
The keyword can consist of one or more words.

Don't forget, that you should never guess whatever was already guessed before! You should always try new options to gain more information!
Only output the same guess if there might be another accepted phrasing of the same keyword (which is usually the case when there are several words).
{secret_message}
Format your answer like this: First, reason about your decision for 2-4 sentences. Summarize what you know, summarize space of possibilities, etc.
Then, your last line should be the keyword that you want to guess.
Don't output anything else. Only output the keyword that you want to guess. Don't say anything like "My guess is {{keyword}}" - always just "{{keyword}}"
"""


async def guess(
    client,
    game: Game,
    include_answer: bool = False
) -> tuple[str, str]:
    response = await call_gpt4(
        client,
        get_question_guess_prompt(game, include_answer),
        game_state=game,
        request_type="guess" + ("_with_answer" if include_answer else ""),
    )
    lines = response.strip().split("\n")
    reasoning = " ".join(lines[:-1])
    guess_str = lines[-1].lower().strip()
    return reasoning, "".join([c for c in guess_str if c.isalnum() or c.isspace() or c == "-"])
