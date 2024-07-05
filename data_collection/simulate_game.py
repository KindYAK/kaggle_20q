from data_collection.answer import answer
from data_collection.guess import guess
from data_collection.models import Game, Round
from data_collection.questions_candidates import get_question_candidates
from data_collection.questions_candidates_entropy import get_question_candidates_scored
from data_collection.questions_pick import pick_question


async def run_game(
    client,
    keyword: str,
):
    game = Game(keyword=keyword)
    for i in range(20):
        print(f"Game {keyword}, round {i}")
        candidates = await get_question_candidates(client, game)
        candidates_scored = await get_question_candidates_scored(client, game, candidates)
        reasoning, question = await pick_question(client, game, candidates_scored)
        reasoning, answer_str = await answer(client, game, keyword, question)
        game.rounds.append(
            Round(
                question=question,
                answer=answer_str,
            )
        )
        reasoning, guessed_keyword = await guess(client, game)
        game.rounds[-1].guess = guessed_keyword
        if guessed_keyword == keyword:
            game.win = True
            return
    game.win = False
    return game
