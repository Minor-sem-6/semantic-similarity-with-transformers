def clean_text(text):

    if isinstance(text, str):
        text = text.lower().strip()
        text = " ".join(text.split())

    return text