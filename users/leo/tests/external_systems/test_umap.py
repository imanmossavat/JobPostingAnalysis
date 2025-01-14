import numpy as np
import pytest
import plotly.express as px
from src.external_systems.umap import UMAPAlg


def test_umap_initialization():
    umap = UMAPAlg()
    assert umap.n_components == 2
    assert umap.random_state is None

    umap_custom = UMAPAlg(n_components=3, random_state=42)
    assert umap_custom.n_components == 3
    assert umap_custom.random_state == 42


def test_reduce_dimensions_different_sizes():
    umap = UMAPAlg()

    # Test with small dataset
    small_data = np.random.rand(20, 5)
    small_reduced = umap.reduce_dimensions(small_data)
    assert small_reduced.shape == (20, 2)

    # Test with larger dataset
    large_data = np.random.rand(100, 5)
    large_reduced = umap.reduce_dimensions(large_data)
    assert large_reduced.shape == (100, 2)


def test_reduce_dimensions_reproducibility():
    data = np.random.rand(100, 5)

    umap1 = UMAPAlg(random_state=42)
    umap2 = UMAPAlg(random_state=42)

    result1 = umap1.reduce_dimensions(data)
    result2 = umap2.reduce_dimensions(data)

    np.testing.assert_array_almost_equal(result1, result2)


def test_generate_2d_viz_output():
    umap = UMAPAlg()
    data = np.random.rand(100, 5)
    reduced_data = umap.reduce_dimensions(data)

    viz = umap.generate_2d_viz(reduced_data)
    assert viz.data[0].x.shape[0] == 100
    assert viz.data[0].y.shape[0] == 100
