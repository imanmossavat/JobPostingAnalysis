import pytest
from src.requests.search_posts import build_search_posts_request

def test_build_search_posts_request_without_parameters():
    request = build_search_posts_request()
    assert request.filters is None
    assert bool(request) is True


def test_build_search_posts_request_with_empty_filters():
    request = build_search_posts_request({})
    assert request.filters == {}
    assert bool(request) is True


def test_build_search_posts_request_with_invalid_filters_parameter():
    request = build_search_posts_request(filters="invalid")
    assert request.has_errors()
    assert request.errors[0]["parameter"] == "filters"
    assert bool(request) is False


def test_build_search_posts_request_with_incorrect_filter_keys():
    request = build_search_posts_request(filters={"invalid": "value"})
    assert request.has_errors()
    assert request.errors[0]["parameter"] == "filters"
    assert bool(request) is False

@pytest.mark.parametrize(
    "key", ["keyword_search", "semantic_search"]
)
def test_build_search_posts_request_with_valid_filters(key):
    filters = {key: dict()}
    request = build_search_posts_request(filters=filters)
    assert request.filters == filters
    assert bool(request) is True
