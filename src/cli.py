from data_preprocessing.cleaner import preprocess_text
from repo_data_parser.get_repo_data import get_repo_info, is_github_repo_url


def get_user_repo() -> str:
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
            "английском языке)"
        )
    else:
        description = preprocess_text[repo_data["description"]]

    repo_data["description_cleaned"] = preprocess_text(description)
    # не спрашивайте, почему ретерн такой, поменяется еще 10 раз
    return repo_data["description_cleaned"]


if __name__ == "__main__":
    get_user_repo()
