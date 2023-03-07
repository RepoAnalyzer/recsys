import pytest

from repo_data_parser.get_repo_data import (
    extract_fields_from_repo_url,
    get_repo_info,
    is_github_repo_url,
)

simple_link = "https://github.com/Nbslab/IU5_ML"


@pytest.mark.parametrize("url, result", [(simple_link, True)])
def test_is_github_repo_url(url, result):
    assert is_github_repo_url(url) == result


@pytest.mark.parametrize(
    "url, values", [(simple_link, {"project": "IU5_ML", "owner": "Nbslab"})]
)
def test_extract_fields_from_repo_url(url, values):
    assert extract_fields_from_repo_url(url) == values


@pytest.mark.parametrize(
    "url, data",
    [
        (
            simple_link,
            {
                "project": "IU5_ML",
                "owner": "Nbslab",
                "language": "Jupyter Notebook",
                "stars": 0,
                "watchers": 0,
                "forks": 0,
                "issue": 0,
                "pr": 0,
                "description": "Machine Learning course from IU5 team",
                "commits": 15,
                "topics": "",
            },
        )
    ],
)
def test_get_repo_info(url, data):
    assert get_repo_info(url) == data
