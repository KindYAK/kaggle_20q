def generate_answer(
    template,
    tokenizer,
    model,
    id_eot
):
    inp_ids = tokenizer(template, return_tensors="pt").to("cuda")
    out_ids = model.generate(**inp_ids, max_new_tokens=15).squeeze()
    start_gen = inp_ids.input_ids.shape[1]
    out_ids = out_ids[start_gen:]
    if id_eot in out_ids:
        stop = out_ids.tolist().index(id_eot)
        out = tokenizer.decode(out_ids[:stop])
    else:
        out = tokenizer.decode(out_ids)
    return out
