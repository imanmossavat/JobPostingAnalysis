import numpy as np
import pytest
from plotly import express as px
from unittest import mock

from src.services.visualize_embeddings import visualize_embeddings


@pytest.fixture
def vector_2d():
    return np.random.rand(100, 2)


"""def test_visualize_embeddings_without_parameters():
    viz = visualize_embeddings()
    assert bool(viz) is False"""


def test_visualize_embeddings_with_parameters(vector_2d):
    umap = mock.Mock()
    reduced_data = umap.reduced_dimensions.return_value = vector_2d
    expected_viz = umap.generate_2d_viz.return_value = px.scatter(
        x=vector_2d[:, 0], y=vector_2d[:, 1]
    )
    embeddings = np.random.rand(100, 100)
    actual_viz = visualize_embeddings(embeddings, umap)
    assert actual_viz == expected_viz
