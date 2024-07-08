import asyncio
import random

from data_collection.answer import answer
from data_collection.guess import guess
from data_collection.models import Game, Round
from data_collection.questions_candidates import get_question_candidates
from data_collection.questions_pick import pick_question


async def run_game(
    client,
    keyword: str,
    include_answer: bool = False,
):
    game = Game(keyword=keyword)
    include_answer_after = random.randint(8, 13)
    for i in range(20):
        print(f"Game {keyword}, round {i}")
        if include_answer and i == include_answer_after:
            print("!HINTING!")
        candidates = await get_question_candidates(client, game, include_answer=include_answer and i >= include_answer_after)
        reasoning, question = await pick_question(client, game, candidates, include_answer=include_answer and i >= include_answer_after)
        reasoning, answer_str = await answer(client, game, keyword, question)
        game.rounds.append(
            Round(
                question=question,
                answer=answer_str,
            )
        )
        reasoning, guessed_keyword = await guess(client, game, include_answer=False)
        game.rounds[-1].guess = guessed_keyword
        if guessed_keyword == keyword:
            game.win = True
            return game
    game.win = False
    return game


async def run_games(
    client,
    keywords: list[str],
    include_answer: bool = False,
):
    games = await asyncio.gather(
        *[run_game(client, keyword, include_answer) for keyword in keywords]
    )
    return games
