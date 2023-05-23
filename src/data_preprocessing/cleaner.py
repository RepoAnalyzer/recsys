import re


def preprocess_text(text):
    # remove digits and spl characters
    text = re.sub(r"[^a-zA-Z#]", " ", str(text))
    # convert to lower case
    text = text.lower()
    # remove user handle
    text = re.sub(r"@[\w]*", "", text)
    # remove http links
    text = re.sub(r"http\S+", "", text)
    # remove additional spaces
    text = re.sub(r"\s+", " ", text)
    return text
