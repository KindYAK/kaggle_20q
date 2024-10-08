import re
import traceback
from itertools import zip_longest

import torch


def generate_answer(
    template,
    tokenizer,
    model,
    id_eot,
    max_new_tokens: int = 75,
    system_prompt: str | None = None,
):
    if "InternLM2ForCausalLM" in str(type(model)):
        return generate_answer_internlm(template, tokenizer, model, system_prompt, max_new_tokens)
    inp_ids = tokenizer(template, return_tensors="pt").to("cuda")
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        id_eot,
    ]
    generation_config = {
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.75,
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "repetition_penalty": 1.0, # 1.0 is no penalty
        "remove_invalid_values": True,
        "eos_token_id": terminators,
        "pad_token_id": id_eot,
        "forced_eos_token_id": id_eot,
        "use_cache": True,
        "no_repeat_ngram_size": 0,
        "num_return_sequences": 1,
    }
    try:
        out_ids = model.generate(
            **inp_ids,
            **generation_config,
        ).squeeze()
    except Exception as e:
        print("! Exception while generating!", e)
        print("".join(traceback.format_exception(None, e, e.__traceback__)))
        generation_config["max_new_tokens"] = 50
        generation_config["use_cache"] = False
        try:
            out_ids = model.generate(
                **inp_ids,
                **generation_config,
            ).squeeze()
        except Exception as e:
            print("! Exception while generating retry!!", e)
            print("".join(traceback.format_exception(None, e, e.__traceback__)))
            return "no"
    start_gen = inp_ids.input_ids.shape[1]
    out_ids = out_ids[start_gen:]
    if id_eot in out_ids:
        stop = out_ids.tolist().index(id_eot)
        out = tokenizer.decode(out_ids[:stop])
    else:
        out = tokenizer.decode(out_ids)
    return out.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()


def generate_answers_batch(
    templates,
    tokenizer,
    model,
    id_eot,
    max_new_tokens: int = 75,
):
    from transformers.utils import logging
    logging.set_verbosity_info()
    input_ids_list = []
    attention_masks_list = []
    valid_indices = []
    for idx, template in enumerate(templates):
        if template is not None:
            tokenized = tokenizer(template, return_tensors="pt")
            inp_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            input_ids_list.append(inp_ids.squeeze(0))
            attention_masks_list.append(attention_mask.squeeze(0))
            valid_indices.append(idx)

    if not input_ids_list:
        return [None] * len(templates)

    max_length = max(input_ids.shape[0] for input_ids in input_ids_list)
    input_ids_padded = []
    attention_masks_padded = []
    for input_ids, attention_mask in zip(input_ids_list, attention_masks_list):
        pad_length = max_length - input_ids.shape[0]
        padded_input_ids = torch.cat([torch.full((pad_length,), 128001), input_ids]) # This is <|end_of_text|>
        input_ids_padded.append(padded_input_ids)

        padded_attention_mask = torch.cat([torch.zeros(pad_length), attention_mask])
        attention_masks_padded.append(padded_attention_mask)

    input_ids_padded = torch.stack(input_ids_padded)
    attention_masks_padded = torch.stack(attention_masks_padded)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        id_eot,
    ]
    generation_config = {
        "do_sample": True,
        "temperature": 0.5,
        "top_p": 0.75,
        "max_new_tokens": max_new_tokens,
        "num_beams": 1,
        "repetition_penalty": 1.0, # 1.0 is no penalty
        "remove_invalid_values": True,
        "eos_token_id": terminators,
        "pad_token_id": 128001,
        "forced_eos_token_id": id_eot,
        "use_cache": True,
        "no_repeat_ngram_size": 0,
        "num_return_sequences": 1,
    }
    out_ids_all = model.generate(
        input_ids_padded.to("cuda"),
        attention_mask=attention_masks_padded.to("cuda"),
        **generation_config,
    )

    outputs = [None] * len(templates)
    for out_ids, idx in zip(out_ids_all, valid_indices):
        response_tokens = out_ids[input_ids_padded.shape[1]:]
        response_tokens = response_tokens[:(response_tokens != 0).nonzero(as_tuple=True)[0][-1] + 1]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True)
        outputs[idx] = response.replace("<|start_header_id|>assistant<|end_header_id|>", "").strip()
    return outputs


def get_qa_history_prompt(obs, include_guesses=False):
    question_answers = ""
    for question, answer, guess in zip_longest(obs['questions'], obs['answers'], obs['guesses']):
        if not answer:
            continue
        if include_guesses and guess:
            question_answers += f"Question: {question} Answer: {answer} Your guess: {guess}\n"
        else:
            question_answers += f"Question: {question} Answer: {answer}\n"

    if question_answers:
        question_answers = f"""Here are the previous questions and answers:
    {question_answers}"""
    return question_answers


def extract_user_prompt(chat_template):
    return chat_template.split("<|start_header_id|>user<|end_header_id|>")[1].split("<|eot_id|>")[0]


def generate_answer_internlm(
    template,
    tokenizer,
    model,
    system_prompt,
    max_new_tokens,
    temperature=0.5,
    top_p=0.75,
):
    response, history = model.chat(tokenizer, extract_user_prompt(template), max_new_tokens=max_new_tokens, temperature=temperature,
                                   meta_instruction=system_prompt, top_p=top_p)
    return response


# SYSTEM_PROMPT_ASKER = """You are a professional, competitive and extremely smart "20 Questions" player.
#
# The rules of the game are as follows:
# - One player thinks of a keyword, and the other player iteratively asks yes-or-no questions to guess the object.
# - The keyword can be a place (country/city/landmark), a thing (any non-living or living object in the physical world), or something else (concept/material/event/activity/...). However, the keyword is never a person!
# - Your goal is to ask smart questions that narrow down the search space and guess the keyword as quickly as possible.
# - Once you've guessed the keyword, the game stops. Hence, there is no reason to guess the same keyword more than once.
# - You have to guess the keyword exactly, letter by letter. Keyword can consist of one or more words.
#
# Keep in mind, that the questions will only be answered with "yes" or "no" (no "I don't know" or "maybe" answers).
# Keep in mind, that questions will be answered by a small LLM, so aim for simpler questions, and expect some errors in the answers.
#
# You have to do your best: always try to play safe, aiming to split space of possibilities in half with each question.
# If there are only a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.
# Be creative. Try to come up with simple, informative questions, don't repeat yourself, always take previous questions and answers into consideration.
# When guessing, don't repeat yourself, and try to account for possible errors and ambiguities in the answers.
#
# You are an extremely smart experienced "20 Questions" player capable of winning most of your matches, balancing risk-taking and safety, and being extremely creative about your questions, while keeping them simple.
# """

SYSTEM_PROMPT_ASKER = """You are an extremely smart experienced "20 Questions" player capable of winning most of your matches, balancing risk-taking and safety, and being extremely creative about your questions, while keeping them simple.

The rules of the game are as follows:
- One player thinks of a keyword, and the other player iteratively asks yes-or-no ((no "I don't know" or "maybe" answers) questions to guess the object.
- The keyword is usually a thing (any non-living or living object in the physical world). However, the keyword is never a person or a place (country/city/landmark/etc)!

Keep in mind, that questions will be answered by a small LLM, so aim for simpler questions, and expect some errors in the answers.

You have to do your best: always try to play safe, aiming to split space of possibilities in half with each question.
If there are only a few questions left, you can start acting more aggressively, intuitively checking some less probable, but high-reward hypotheses.
"""

SYSTEM_PROMPT_ANSWERER = """You are a professional, ethical and extremely smart "20 Questions" player.

The rules of the game are as follows:
- One player thinks of a keyword, and the other player iteratively asks yes-or-no questions to guess the object.

Keep in mind, that the questions will only be answered with "yes" or "no" (no "I don't know" or "maybe" answers).

Your goal is to answer questions as truthfully as possible. Even if answer is not "yes" or "no" aim to be helpful and truthful.
"""
