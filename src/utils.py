def apply_mapping(data):
    """
    Maps the keys and roles from the Orca dataset to that expected by the model
    """
    KEY_MAPPING = {"from": "role","value": "content"}
    ROLE_MAPPING = {"human": "user", "gpt": "assistant"}
    result = []
    for item in data:
        new_item = {}
        for key, value in item.items():
            new_key = KEY_MAPPING.get(key, key)
            new_value = ROLE_MAPPING.get(value, value) if new_key == "role" else value
            new_item[new_key] = new_value
        result.append(new_item)
    return result