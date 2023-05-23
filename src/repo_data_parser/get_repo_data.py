import re

import github
from github import Github


def is_github_repo_url(url: str) -> bool:
    """
    Used to remove markdown anchor links, website references... and leave
    only github repositories. It also takes anchors into consideration. For
    example, https://github.com/eleventigers/awesome-rxjava#readme
    See [available symbols in github repository name](shorturl.at/ACFG0).
    """
    if not bool(
        re.compile(r"^https?://github.com/[\w.-]+/[\w.-]+(#[\w.-]+)?").fullmatch(url)
    ):
        return False
    else:
        g = Github("")
        result = extract_fields_from_repo_url(url)
        try:
            g.get_repo(f"{result['owner']}/{result['name']}")
        except github.GithubException.UnknownObjectException:
            print(
                "Такого репозитория не существует, проверьте правильность "
                "вводимых данных"
            )
            return False
        else:
            return True


def extract_fields_from_repo_url(repository_url: str) -> dict:
    """
    Used to extract name of repository and name of the User owner of the repository

    :param repository_url: url of the github repository
    :type repository_url: str
    :return: dict with repo name, name of the owner of the repo, and full_name of repo
    """
    if repository_url.endswith(".git"):
        repository_url = repository_url.replace(".git", "")
    if matches := re.search(r"([a-z]+):\/\/([^/]*)\/([^/]*)\/(.*)", repository_url):
        name_space = str(matches.group(3))
        project = str(matches.group(4))

    return {"name": project, "owner": name_space}


def get_repo_info(url: str) -> dict:
    """
    Used to request data from github repository and group it in dict

    :param url: url of the github repository
    :type url: str

    :return: dict with data gained from requests
    """

    if not is_github_repo_url(url):
        raise ValueError("No GitHub url provided")
    result = extract_fields_from_repo_url(url)
    g = Github("")
    repo = g.get_repo(f"{result['owner']}/{result['name']}")
    result["full_name"] = repo.full_name
    result["language"] = repo.language
    result["stars"] = repo.stargazers_count
    result["watchers"] = repo.watchers_count
    result["forks"] = repo.forks_count
    result["open_issues"] = repo.get_issues().totalCount
    result["description"] = repo.description
    result["commits"] = repo.get_commits().totalCount
    result["topics"] = ", ".join(repo.get_topics())
    return result
