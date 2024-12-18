import pandas as pd
from src.interfaces.repository import Repository


class ModelsDfRepo(Repository):
    def __init__(self, models_df: pd.DataFrame):
        self.models_df = models_df

    def list(self, filters):
        return self.models_df.loc[
            self.models_df["id"] == filters.get("model_id"), "name"
        ][0]
