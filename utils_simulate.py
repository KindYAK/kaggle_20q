import datetime

from more_itertools import chunked

from data_collection.models import Game, Round
from utils_answerer import answer_batch
from utils_asker import ask_batch
from utils_guesser import guess_batch


def _game_to_obs_dict(game: Game):
    return {
        'remainingOverageTime': 300,
        'questions': [round_.question for round_ in game.rounds],
        'guesses': [round_.guess for round_ in game.rounds if round_.guess],
        'answers': [round_.answer for round_ in game.rounds if round_.answer],
        'role': None,
        'turnType': None,
        'keyword': game.keyword if game.keyword is not None else 'None',
        'category': None,
        'step': len(game.rounds) * 2 + 20
    }


def _run_games(
    keywords: list[str],
    tokenizer,
    model,
    id_eot,
):
    from transformers.utils import logging
    logging.set_verbosity_info()
    games = [Game(keyword=keyword) for keyword in keywords]
    for i in range(20):
        # print("! Running round", i, datetime.datetime.now())
        questions = ask_batch(
            obs_list=[_game_to_obs_dict(game) if game.win is None else None for game in games],
            model=model,
            tokenizer=tokenizer,
            id_eot=id_eot,
        )
        for game, question in zip(games, questions):
            game.rounds.append(
                Round(
                    question=question,
                )
            )

        answers = answer_batch(
            obs_list=[_game_to_obs_dict(game) if game.win is None else None for game in games],
            model=model,
            tokenizer=tokenizer,
            id_eot=id_eot,
        )
        for game, answer in zip(games, answers):
            game.rounds[-1].answer = answer

        guesses = guess_batch(
            obs_list=[_game_to_obs_dict(game) if game.win is None else None for game in games],
            model=model,
            tokenizer=tokenizer,
            id_eot=id_eot,
        )
        for game, guessed_keyword in zip(games, guesses):
            def _clean_kw(x):
                return "".join([c for c in x if c.isalpha()]).lower()
            game.rounds[-1].guess = guessed_keyword
            if _clean_kw(guessed_keyword) == _clean_kw(game.keyword):
                game.win = True

    for game in games:
        if game.win is None:
            game.win = False
    return games


def run_games(
    keywords: list[str],
    tokenizer,
    model,
    id_eot,
    batch_size: int = 10,
):
    results = []
    for batch in chunked(keywords, batch_size):
        print("Processing batch of games")
        batch_results = _run_games(
            keywords=batch,
            tokenizer=tokenizer,
            model=model,
            id_eot=id_eot,
        )
        results.extend(batch_results)
    return results
