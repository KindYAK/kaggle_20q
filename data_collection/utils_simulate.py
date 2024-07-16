import datetime

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


def run_games(
    keywords: list[str],
    tokenizer,
    model,
    id_eot,
):
    games = [Game(keyword=keyword) for keyword in keywords]
    for i in range(20):
        print("! Running round", i, datetime.datetime.now())
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
            game.rounds[-1].guess = guessed_keyword
            if guessed_keyword == game.keyword:
                game.win = True

    for game in games:
        if game.win is None:
            game.win = False
    return games
