def tokenize_from_chars(words):
    chars = sorted(list(set(''.join(words))))
    string_to_token_id = {ch:i+1 for i, ch in enumerate(chars)}
    string_to_token_id['.'] = 0
    token_id_to_string = {i:ch for ch,i in string_to_token_id.items()}
    return string_to_token_id, token_id_to_string