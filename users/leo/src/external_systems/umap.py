from typing import Optional
import plotly.express as px
from umap import UMAP


class UMAPAlg:
    def __init__(
        self,
        n_components: int = 2,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.model = UMAP(n_components=n_components, random_state=random_state)

    def reduce_dimensions(self, data):
        return self.model.fit_transform(data)

    def generate_2d_viz(self, data):
        return px.scatter(x=data[:, 0], y=data[:, 1])
