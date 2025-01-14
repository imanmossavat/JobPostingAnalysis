def visualize_embeddings(embeddings, dim_reduction):
    """if not embeddings:
    return False"""
    reduced_data = dim_reduction.reduce_dimensions(embeddings)
    return dim_reduction.generate_2d_viz(reduced_data)
