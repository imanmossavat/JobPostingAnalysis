from collections.abc import Mapping


class PostsSearchInvalidRequest:
    def __init__(self):
        self.errors = []

    def add_error(self, parameter, message):
        self.errors.append({"parameter": parameter, "message": message})
    
    def has_errors(self):
        return len(self.errors) > 0
    
    def __bool__(self):
        return False


class PostsSearchValidRequest:
    def __init__(self, filters):
        self.filters = filters

    def __bool__(self):
        return True


def build_search_posts_request(filters=None):
    accepted_filters = ["keyword_search", "semantic_search"]
    invalid_request = PostsSearchInvalidRequest()

    if filters is not None:
        if not isinstance(filters, Mapping):
            invalid_request.add_error("filters", "Is not iterable")
            return invalid_request

        for key in filters.keys():
            if key not in accepted_filters:
                invalid_request.add_error(
                    "filters", "Key {} cannot be used".format(key)
                )
        if invalid_request.has_errors():
            return invalid_request
        
    return PostsSearchValidRequest(filters=filters)
    