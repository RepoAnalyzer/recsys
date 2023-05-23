import re
from copy import deepcopy

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


def get_user_repo():
    is_passed = False
    url = "user_url"
    url = input("Пожалуйста, введите валидный URL GitHub репозитория: ")
    for i in range(4):
        if not is_github_repo_url(url):
            print(
                "Полученный ресурс не является GitHub репозиторием, пожалуйста, "
                "проверьте правильность вводимых "
                "данных и попробуйте снова"
            )
            url = input(
                f"Пожалуйста, введите валидный URL GitHub репозитория "
                f"(Осталось попыток: {4-i}): "
            )
        else:
            is_passed = True
            break
    if is_passed:
        pass
    else:
        return None

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

    if bool(re.search("[a-zA-Z]", preprocess_text(description))):
        repo_data["description_cleaned"] = preprocess_text(description)
    else:
        while not bool(re.search("[a-zA-Z]", preprocess_text(description))):
            description = input(
                "Полученное описание не валидно, пожалуйста, введите новое описание: \n"
            )
        repo_data["description_cleaned"] = preprocess_text(description)
    return repo_data


def change_repo_params(repo: dict):
    repo_input = deepcopy(repo)
    print("Параметры полученного репозитория")
    for param in repo_input.keys():
        print(f"{param}: {repo_input[param]}")
    answer_del = input("Вы хотите удалить данный репозиторий? (Да/Нет): ").lower()
    for i in range(4):
        is_passed = False
        if answer_del not in ["да", "нет"]:
            print("Пожалуйста, введите ответ в заданном формате")
            answer_del = input(
                "Вы хотите удалить данный репозиторий? (Да/Нет) "
                f"(Осталось попыток: {4-i}): "
            ).lower()
        else:
            is_passed = True
            break
    if is_passed:
        pass
    else:
        return None
    if answer_del == "да":
        del repo_input
        del repo
        print("Репозиторий удален из анализа")
        return None
    elif answer_del == "нет":
        pass
    # else:
    answer_change = input(
        "Вы хотите изменить описание данного репозитория? (Да/Нет): "
    ).lower()
    for i in range(4):
        is_passed = False
        if answer_change not in ["да", "нет"]:
            print("Пожалуйста, введите ответ в заданном формате")
            answer_change = input(
                f"Вы хотите изменить описание данного репозитория? (Да/Нет) "
                f"(Осталось попыток: {4-i}): "
            ).lower()
        else:
            is_passed = True
            break
    if is_passed:
        pass
    else:
        return None
    if answer_change == "да":
        description = input(
            "Введите новое описание репозитория (на английском языке): \n"
        )
        if bool(re.search("[a-zA-Z]", preprocess_text(description))):
            repo_input["description"] = description
            repo_input["description_cleaned"] = preprocess_text(description)
        else:
            while not bool(re.search("[a-zA-Z]", preprocess_text(description))):
                description = input(
                    "Полученное описание не валидно, пожалуйста, "
                    "введите новое описание: \n"
                )
            repo_input["description"] = description
            repo_input["description_cleaned"] = preprocess_text(description)
    elif answer_change == "нет":
        pass
    return repo_input


def get_nns_by_repo_id(data, dim, repo_id, n_nn=10):
    repos_ann = AnnoyIndex(dim, "angular")
    repos_ann.load("models/lda/repos.ann")  # super fast, will just mmap the file
    repos_mapping = pd.read_pickle("models/lda/repos_mapping.pkl")
    repos_mapping_rev = {v: k for k, v in repos_mapping.items()}
    try:
        repos_mapping[repo_id]
    except KeyError:
        repos_mapping[repo_id] = repo_id
        repos_mapping_rev[repo_id] = repo_id
    else:
        pass

    res, dist = repos_ann.get_nns_by_item(
        repos_mapping[repo_id], n_nn, include_distances=True
    )  # will find the n nearest neighbors
    res = [repos_mapping_rev[r] for r in res]
    res[:] = [x for x in res if x <= data.shape[0]]
    res = data.loc[res[:5]]
    dist = dist[:5]
    res["distance"] = [round(d, 4) for d in dist]
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


def remap(dim: int, repos_df: pd.DataFrame, n_trees: int = 1_000):
    repos_ann = AnnoyIndex(dim, "angular")  # Length of item vector that will be indexed
    repos_mapping = {}
    for i, (repository_id, e) in enumerate(repos_df["embeddings_lda"].iteritems()):
        repos_ann.add_item(repository_id, list(e))
        repos_mapping[repository_id] = i
    n_trees = n_trees
    repos_ann.build(n_trees)
    repos_ann.save("models/lda/repos-1.ann")
    pd.to_pickle(repos_mapping, "models/lda/repos_mapping-1.pkl")
    repos_df.to_csv("recsys/data/repositories_embeddings_LDA_3.csv", index=False)


def find_user_repo_in_db(full_name: str, repo_data):
    count_vect = joblib.load("models/lda/countVect.pkl")
    lda = joblib.load("models/lda/lda.pkl")
    repo_dict = deepcopy(repo_data)
    repo_dict["embeddings_lda"] = create_embeddings(
        repo_dict["description_cleaned"], lda, count_vect
    )
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
        data.index[data["full_name"] == full_name][0]
    except IndexError:
        print("Такого репозитория нет в базе, время работы системы увеличено")
        data = data.append(repo_dict, ignore_index=True)
        new_df = data.copy()
        id = new_df.index[new_df["full_name"] == full_name][0]
        print(id, new_df["embeddings_lda"])
        # remap(len(repo_dict["embeddings_lda"]), new_df)
        id = None
    else:
        id = data.index[data["full_name"] == full_name][0]

    dim = len(data.embeddings_lda.iloc[0])
    return id, data, dim


if __name__ == "__main__":
    is_passed = False
    while not is_passed:
        repo = get_user_repo()
        repo_changed = change_repo_params(repo=repo)
        if type(repo_changed) is dict:
            is_passed = True
    repo_id, data, dim = find_user_repo_in_db(
        repo_changed["full_name"], repo_data=repo_changed
    )
    if repo_id is not None:
        result_df = get_nns_by_repo_id(data, dim, repo_id)
        print("Список похожих репозиториев: ")
        print(result_df)
        print("Сохранение...")
        result_df.to_csv("../results/result.csv", index=False)
        print("Результат сохранен успешно")

# https://github.com/Nbslab/BCIT_3sem

# https://github.com/mrdoob/three.js
# https://github.com/aframevr/aframe
