import numpy as np


def create_embeddings(text: str, lda, count_vect) -> np.ndarray:
    term_doc_matrix_for_one = count_vect.transform([text])
    embeddings = lda.transform(term_doc_matrix_for_one)
    return embeddings


if __name__ == "__main__":
    create_embeddings("simple javascript library")
