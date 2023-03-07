import re

from github import Github


def is_github_repo_url(url: str) -> bool:
    """
    Used to remove markdown anchor links, website references... and leave
    only github repositories. It also takes anchors into consideration. For
    example, https://github.com/eleventigers/awesome-rxjava#readme
    See [available symbols in github repository name](shorturl.at/ACFG0).
    """
    return bool(
        re.compile(r"^https?://github.com/[\w.-]+/[\w.-]+(#[\w.-]+)?").fullmatch(url)
    )


def extract_fields_from_repo_url(repository_url: str) -> dict:
    """Used to extract name of repository and name of the User owner of the repository

    Parameters
    ----------
    repository_url : str
        url of the github repository

    Returns
    -------
    dict
        dict with repo name, name of the owner of the repo, and full_name of repo
    """
    if repository_url.endswith(".git"):
        repository_url = repository_url.replace(".git", "")
    matches = re.search(r"([a-z]+):\/\/([^/]*)\/([^/]*)\/(.*)", repository_url)
    if matches := re.search(r"([a-z]+):\/\/([^/]*)\/([^/]*)\/(.*)", repository_url):
        name_space = str(matches.group(3))
        project = str(matches.group(4))

    return {"project": project, "owner": name_space}


def get_repo_info(url: str) -> dict:
    """Used to request data from github repository and group it in dict

    Parameters
    ----------
    repo_fullname : str
        fullname of the repo in format {owner}/{repo-name}

    Returns
    -------
    dict
        dict with data gained from requests
    """

    if is_github_repo_url(url):
        result = extract_fields_from_repo_url(url)
        g = Github()
        repo = g.get_repo(f"{result['owner']}/{result['project']}")
        result["language"] = repo.language
        result["stars"] = repo.stargazers_count
        result["watchers"] = repo.watchers_count
        result["forks"] = repo.forks_count
        result["issue"] = repo.get_issues().totalCount
        result["pr"] = repo.get_pulls().totalCount
        result["description"] = repo.description
        result["commits"] = repo.get_commits().totalCount
        result["topics"] = ", ".join(repo.get_topics())
        return result
    else:
        raise ValueError("Not GitHub url provided")
