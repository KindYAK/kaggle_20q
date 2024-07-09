from data_collection.models import Game
from data_collection.utils import get_qa_history_prompt, call_gpt4, get_secret_message


def get_generate_snapshot_prompt(
    keyword: str,
):
    return f"""You are an expert "20 Questions" player.
    
I need you to help me generate educational content for intermediate players.
I will present you with a keyword, and you will generate a sequence of questions, answers and guesses.

Example:
Question: Is it a living thing? Answer: no Your guess: doorbell
Question: Is it a physical object? Answer: yes Your guess: computer
Question: Is it man-made? Answer: yes Your guess: car
Question: Is it used in everyday life? Answer: yes Your guess: phone
...

You need to:
1) Generate 9-19 question-answers that slowly converge to the correct keyword
2) Never give away that you know the answer. Make the sequence look realistic and by educational.
3) Still, make smart, safe moves every round - try to divide space of possibilities in half every time, but if you are running out of questions (20 is max) - you can act a bit more risky.
4) Finally, you should arrive at the correct answer. But never by pure luck, always methodically, sometimes with a bit riskier moves.
5) You can sometimes use questions like "Does the keyword consist of more than one word?". You can sometimes ask things like "Does the keyword start with letter A?". However, don't overdo it, and only if it makes sense.
6) The game ends when the keyword is guessed. So, it doesn't have to last for 20 questions, but should finish in 9-19 questions.

Your keyword is: {keyword}

Now, output the sequence in the format presented above. Nothing else. Follow the format in the example: every round in a single line, not additional line breaks. Guess is onl
"""


async def generate_snapshot(
    client,
    keyword: str,
):
    response = await call_gpt4(
        client,
        get_generate_snapshot_prompt(keyword),
    )
    return generate_dataset_samples(response, keyword)


def generate_dataset_samples(
    generated_sequence: str,
    keyword: str,
):
    samples = []
    samples.append({
        'prompt': f"""You are playing "20 Questions" game.
    
Here are the previous questions and answers:
{generated_sequence}

You goal is to make a guess about the keyword given the history of questions and answers.
History above contains previous guesses. When the guess is correct, the game stops. So, there's not reason to try a guess that was already made!
Also, keep in mind that guess only counts if it's correct letter-by-letter. So, if you have strong reasons to believe that one of the previous guesses is correct, but was
phrased differently - you can try to guess it again. 
Don't forget that some inconsistencies in answers can be caused by errors in the answering LLM.
The keyword can consist of one or more words.

Don't forget, that you should never guess whatever was already guessed before! You should always try new options to gain more information!
Only output the same guess if there might be another accepted phrasing of the same keyword (which is usually the case when there are several words).

Format your answer like this: First, reason about your decision for 2-4 sentences. Summarize what you know, summarize space of possibilities, etc.
Then, your last line should be the keyword that you want to guess.
Don't output anything else. Only output the keyword that you want to guess. Don't say anything like "My guess is {{keyword}}" - always just "{{keyword}}"
""",
        'response': f"""Based on the available questions, answers and given that none of the previous guesses were correct, I think that the best guess would be {keyword}
{keyword}""",
        'type': 'guess_from_snapshot'})
    lines = generated_sequence.strip().split("\n")
    for i in range(1, len(lines)):
        history_before_i = "\n".join(lines[:i])
        samples.append(
            {
                'prompt': f"""You are playing "20 Questions" game.
    
Here are the previous questions and answers:
{history_before_i}
    
You need to generate 1 candidates for the next question to ask.
You should mix safe questions with more creative and non-orthodox ones.
Safe questions are usually ones that split the space of possibilities in half.
If you have a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.
Consider asking non-standard questions like "Does the keyword consist of a single word?, however only do it if you don't know it yet, and if you think that it's important to know at this point in the game.
Consider asking questions like "Does the keyword start with letter 'A'?" or "Does the keyword contain letter 'E'?". However, don't overdo it, focus on more substantial questions.

Format your answer like this: output every question on a new line. Don't output anything else.
""",
                'response': lines[i],
                'type': 'questions_candidates_from_snapshot'
            }
        )
    return samples
