def tokenize_t5_explanation_txt(tokenizer, max_text_length, txt, prompt=False, lm_adapt=False):
    '''
    Tokenizes the text by trimming the appropriate txt
    :param tokenizer:
    :param max_text_length:
    :param txt:
    '''

    txt_input_ids = tokenizer.encode(txt, add_special_tokens=False) + [tokenizer.eos_token_id]

    # Add 2 to account for "explanation: "
    if prompt: 
        if not isinstance(prompt, str):
            prompt = f"explanation: If {tokenizer.additional_special_tokens[0]}, then {tokenizer.additional_special_tokens[1]}"
        additional_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    else:
        start_token = '' if lm_adapt else tokenizer.additional_special_tokens[0]
        prompt = f"explanation: {start_token}"
        additional_tokens = len(tokenizer.encode(prompt, add_special_tokens=True))
    tot_length = len(txt_input_ids) + additional_tokens

    # trim text
    if tot_length > max_text_length:
        num_trim = tot_length - max_text_length
        txt = "example".join(tokenizer.decode(txt_input_ids[:-num_trim]).split("example")[:-1])
    
    # Add "explanation: " and tokenize
    new_txt_input_ids = tokenizer(txt + prompt,
                                  padding="max_length",
                                  max_length=max_text_length,
                                  truncation=True,
                                  return_tensors='pt').input_ids
    trunc_input_ids = new_txt_input_ids
    
    return trunc_input_ids