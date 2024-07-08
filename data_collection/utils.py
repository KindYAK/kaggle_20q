import asyncio
import datetime
import pickle
import logging

import tenacity
from openai import AsyncOpenAI

from data_collection.models import Game
from data_collection.semaphore import get_semaphore

SYSTEM_PROMPT_ASKER = """You are a professional, competitive and extremely smart "20 Questions" player.

The rules of the game are as follows:
- One player thinks of a keyword, and the other player iteratively asks yes-or-no questions to guess the object.
- The keyword can be a place (country/city/landmark), a thing (any non-living or living object in the physical world), or something else (concept/material/event/activity/...). However, the keyword is never a person!
- Your goal is to ask smart questions that narrow down the search space and guess the keyword as quickly as possible.
- Once you've guessed the keyword, the game stops. Hence, there is no reason to guess the same keyword more than once.
- You have to guess the keyword exactly, letter by letter. Keyword can consist of one or more words.

Keep in mind, that the questions will only be answered with "yes" or "no" (no "I don't know" or "maybe" answers).
Keep in mind, that questions will be answered by a small LLM, so aim for simpler questions, and expect some errors in the answers.

You have to do your best: always try to play safe, aiming to split space of possibilities in half with each question.
If there are only a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.
Be creative. Try to come up with simple, informative questions, don't repeat yourself, always take previous questions and answers into consideration.
When guessing, don't repeat yourself, and try to account for possible errors and ambiguities in the answers.

You are an extremely smart experienced "20 Questions" player capable of winning most of your matches, balancing risk-taking and safety, and being extremely creative about your questions, while keeping them simple.
"""

SYSTEM_PROMPT_ANSWERER = """You are a professional, ethical and extremely smart "20 Questions" player.

The rules of the game are as follows:
- One player thinks of a keyword, and the other player iteratively asks yes-or-no questions to guess the object.

Keep in mind, that the questions will only be answered with "yes" or "no" (no "I don't know" or "maybe" answers).

Your goal is to answer questions as truthfully as possible. Even if answer is not "yes" or "no" aim to be helpful and truthful.
"""


def log_before_sleep(retry_state):
    logging.info(f"Retrying: attempt #{retry_state.attempt_number}, waiting {retry_state.next_action.sleep} seconds due to {retry_state.outcome.exception()}")


dataset = []


@tenacity.retry(
    wait=tenacity.wait_fixed(30) + tenacity.wait_exponential(multiplier=1, min=30, max=60),
    stop=tenacity.stop_after_attempt(10),
    retry=tenacity.retry_if_exception_type(Exception),
    before_sleep=log_before_sleep
)
async def call_gpt4(
    client: AsyncOpenAI,
    prompt: str,
    request_type: str,
    game_state: Game,
    system_prompt: str | None = SYSTEM_PROMPT_ASKER,
    # Formatted as a list of dictionaries with keys "role" and "content"
    message_history: list[dict[str, str]] | None = None,
    temperature: float = 0.0,
):
    global dataset

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if message_history:
        messages.extend(message_history)
    messages.append({"role": "user", "content": prompt})
    async with get_semaphore("gpt-4"):
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=temperature,
        )
    response_txt = response.choices[0].message.content
    # Print usage (tokens)
    # print(f"Tokens used: {response.usage.total_tokens}")
    dataset.append(
        {
            "prompt": prompt,
            "response": response_txt,
            "type": request_type,
            "game_state": game_state.copy(deep=True),
        }
    )
    if len(dataset) % 100 == 0:
        # with open(f"dataset_{datetime.datetime.now()}.json", "w") as f:
        #     json.dump(dataset, f)
        dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pickle.dump(dataset, open(f"dataset_{dt}.pkl", "wb"))
    return response_txt


def get_qa_history_prompt(game: Game, include_guesses=False):
    question_answers = ""
    for r in game.rounds:
        if not r.answer:
            continue
        if include_guesses and r.guess:
            question_answers += f"Question: {r.question} Answer: {r.answer} Your guess: {r.guess}\n"
        else:
            question_answers += f"Question: {r.question} Answer: {r.answer}\n"

    if question_answers:
        question_answers = f"""Here are the previous questions and answers:
    {question_answers}"""
    return question_answers
