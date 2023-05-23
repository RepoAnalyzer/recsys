import re

import joblib
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

from data_preprocessing.cleaner import preprocess_text
from data_preprocessing.tokenization import create_embeddings
from repo_data_parser.get_repo_data import get_repo_info, is_github_repo_url


def is_valid_description(text: str) -> bool:
    if text.isdigit():
        return False
    else:
        return True


def get_user_repo() -> dict:
    count_vect = joblib.load("models/lda/countVect.pkl")
    lda = joblib.load("models/lda/lda.pkl")
    url = "user_url"
    while not is_github_repo_url(url):
        url = input("Пожалуйста, введите валидный URL GitHub репозитория: ")
        if not is_github_repo_url(url):
            print(
                "Введенный URL невалиден, пожалуйста, проверьте правильность вводимых "
                "данных и попробуйте снова"
            )

    repo_data = get_repo_info(url)

    print(repo_data)
    if repo_data["description"] is None:
        description = input(
            "В выбранном репозитории отсутствует описание, пожалуйста, "
            "коротко опишите его функционал или область применения(на "
            "английском языке): \n"
        )
    else:
        description = repo_data["description"]

    if re.search("[a-zA-Z]", preprocess_text(description)):
        repo_data["description_cleaned"] = preprocess_text(description)
    else:
        while not re.search("[a-zA-Z]", preprocess_text(description)):
            description = input(
                "Полученное описание не валидно, пожалуйста, введите новое описание: \n"
            )
        repo_data["description_cleaned"] = preprocess_text(description)

    # TODO: предоставить здесь пользователю выбор поменять репу/отредактировать описание

    repo_data["embeddings"] = create_embeddings(
        repo_data["description_cleaned"], lda, count_vect
    )
    return repo_data


def get_nns_by_repo_id(data, dim, repo_id, n_nn=10):
    repos_ann = AnnoyIndex(dim, "angular")
    repos_ann.load("models/lda/repos.ann")  # super fast, will just mmap the file
    repos_mapping = pd.read_pickle("models/lda/repos_mapping.pkl")
    repos_mapping_rev = {v: k for k, v in repos_mapping.items()}
    res, dist = repos_ann.get_nns_by_item(
        repos_mapping[repo_id], n_nn, include_distances=True
    )  # will find the n nearest neighbors
    res = [repos_mapping_rev[r] for r in res]
    res[:] = [x for x in res if x <= data.shape[0]]
    res = data.loc[res[:5]]
    dist = dist[:5]
    res["distance"] = [round(d, 3) for d in dist]
    return res[["full_name", "distance"]]


def convert(row):
    res = []
    for value in row.embeddings_lda:
        try:
            value = float(value)
        except ValueError:
            value = 0.0
        else:
            value = float(value)
        res.append(value)
    return np.array(res)


def find_user_repo_in_db(full_name: str):
    data = pd.read_csv(
        "../data/repositories_embeddings_LDA_2.csv",
        engine="python",
        error_bad_lines=False,
    )
    data.embeddings_lda = data.embeddings_lda.apply(
        lambda s: [
            idx.strip("[]").replace("\n", "").replace(" ", ", ") for idx in s.split(",")
        ]
    )
    data.embeddings_lda = data.embeddings_lda.apply(lambda s: s[0].split(","))
    data.embeddings_lda = data.apply(convert, axis=1)
    try:
        np.where(data.full_name == full_name)[0][0]
    except IndexError:
        print("Такого репозитория нет в базе, время работы системы увеличено")
        id = -1
        # TODO: инициализировать переобучение
    else:
        id = np.where(data.full_name == full_name)[0][0]
    return id


if __name__ == "__main__":
    repo = get_user_repo()
    # find_user_repo_in_db("apache/spark")
# https://github.com/Nbslab/BCIT_3sem
