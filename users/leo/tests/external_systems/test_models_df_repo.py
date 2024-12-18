import pandas as pd
from pandas.testing import assert_frame_equal
from src.external_systems.models_df_repo import ModelsDfRepo


MODELS = pd.DataFrame(
    {
        "id": [1, 2, 3],
        "name": ["model1", "model2", "model3"],
    }
)


def test_repository_list_with_model_id_equal():
    repo = ModelsDfRepo(MODELS)
    filter = {"model_id": 1}
    models_expected = "model1"
    assert repo.list(filter) == models_expected
