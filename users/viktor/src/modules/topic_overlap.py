# topic_overlap_graph_generator.py
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import os
from datetime import datetime
from interfaces import ITopicOverlapGraphGenerator

class TopicOverlapGraphGenerator(ITopicOverlapGraphGenerator):
    def generate_graph(self, keyword_dict: dict, output_subfolder: str, name_of_topics: str, filtered_df: pd.DataFrame = None) -> None:
        """
        Generate and save a graph-based visualization of topic overlaps using circles and edges.
        
        Args:
            keyword_dict: Dictionary where keys are topics and values are lists of keywords.
            output_subfolder: Folder where the graph visualization will be saved.
            name_of_topics: Name of the topics (e.g., "Job").
            filtered_df: Optional filtered DataFrame for the graph.
        
        Returns:
            None
        """
        if not keyword_dict or not isinstance(keyword_dict, dict):
            raise ValueError("keyword_dict must be a non-empty dictionary.")
        
        try:
            # Create a graph
            G = nx.Graph()

            # Add nodes and calculate sizes based on the number of keywords
            node_sizes = {}
            for topic, keywords in keyword_dict.items():
                if not isinstance(keywords, list):
                    raise ValueError(f"Keywords for {topic} should be a list.")
                G.add_node(topic, size=len(keywords))
                node_sizes[topic] = len(keywords)

            # Add edges based on keyword overlap
            topics = list(keyword_dict.keys())
            for i, topic1 in enumerate(topics):
                for j, topic2 in enumerate(topics):
                    if i < j:  # Avoid duplicate edges
                        overlap = len(set(keyword_dict[topic1]) & set(keyword_dict[topic2]))
                        if overlap > 0:  # Only add an edge if there is overlap
                            G.add_edge(topic1, topic2, weight=overlap)

            # Generate positions for nodes using a force-directed layout
            pos = nx.spring_layout(G, seed=42)

            # Assign a unique color to each node
            unique_colors = [plt.cm.tab20(i / len(G.nodes())) for i in range(len(G.nodes()))]
            node_color_map = {node: unique_colors[idx] for idx, node in enumerate(G.nodes())}
            node_colors = [node_color_map[node] for node in G.nodes()]

            # Draw nodes with sizes proportional to their keyword count
            sizes = [node_sizes[node] * 100 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, alpha=0.8, edgecolors='k')

            # Draw edges with thickness proportional to the overlap
            edges = G.edges(data=True)
            edge_widths = [data['weight'] for _, _, data in edges]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)

            # Create a legend for the nodes
            legend_elements = [
                Patch(facecolor=node_color_map[node], edgecolor='k', label=node) for node in G.nodes()
            ]

            # Add a legend for the edges
            max_weight = max(edge_widths) if edge_widths else 1
            edge_legend_elements = [
                Line2D([0], [0], color='gray', lw=max(1, width / max_weight * 5), label=f'Overlap {int(width)}')
                for width in sorted(set(edge_widths), reverse=True)
            ]

            plt.legend(
                handles=legend_elements + edge_legend_elements,
                loc='upper left',
                bbox_to_anchor=(1, 1),
                fontsize=8,
                title="Legend"
            )

            # Add a title
            plt.title(f"{name_of_topics} Overlap Graph", fontsize=14)

            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_filename = f"topic_overlap_graph_{timestamp}.png"  # Include timestamp in the filename
            graph_path = os.path.join(output_subfolder, graph_filename)
            
            # Save the graph
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            print(f"Topic overlap graph saved at {graph_path}")

        except Exception as e:
            print(f"Error generating topic overlap graph: {e}")
        finally:
            plt.close()

''' class TopicOverlapGraphGenerator(ITopicOverlapGraphGenerator):
    def generate_graph(self, keyword_dict: dict, output_subfolder: str, name_of_topics: str, filtered_df: pd.DataFrame = None) -> None:
        """
        Generate and save a graph-based visualization of topic overlaps using circles and edges.
        
        Args:
            keyword_dict: Dictionary where keys are topics and values are lists of keywords.
            output_subfolder: Folder where the graph visualization will be saved.
            name_of_topics: Name of the topics (e.g., "Job").
            filtered_df: Optional filtered DataFrame for the graph.
        
        Returns:
            None
        """
        try:
            # Create a graph
            G = nx.Graph()

            # Add nodes and calculate sizes based on the number of keywords
            node_sizes = {}
            for topic, keywords in keyword_dict.items():
                G.add_node(topic, size=len(keywords))
                node_sizes[topic] = len(keywords)

            # Add edges based on keyword overlap
            topics = list(keyword_dict.keys())
            for i, topic1 in enumerate(topics):
                for j, topic2 in enumerate(topics):
                    if i < j:  # Avoid duplicate edges
                        overlap = len(set(keyword_dict[topic1]) & set(keyword_dict[topic2]))
                        if overlap > 0:  # Only add an edge if there is overlap
                            G.add_edge(topic1, topic2, weight=overlap)

            # Generate positions for nodes using a force-directed layout
            pos = nx.spring_layout(G, seed=42)

            # Assign a unique color to each node
            unique_colors = [plt.cm.tab20(i / len(G.nodes())) for i in range(len(G.nodes()))]
            node_color_map = {node: unique_colors[idx] for idx, node in enumerate(G.nodes())}
            node_colors = [node_color_map[node] for node in G.nodes()]

            # Draw nodes with sizes proportional to their keyword count
            sizes = [node_sizes[node] * 100 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, alpha=0.8, edgecolors='k')

            # Draw edges with thickness proportional to the overlap
            edges = G.edges(data=True)
            edge_widths = [data['weight'] for _, _, data in edges]
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)

            # Create a legend for the nodes
            legend_elements = [
                Patch(facecolor=node_color_map[node], edgecolor='k', label=node) for node in G.nodes()
            ]

            # Add a legend for the edges
            max_weight = max(edge_widths) if edge_widths else 1
            edge_legend_elements = [
                Line2D([0], [0], color='gray', lw=max(1, width / max_weight * 5), label=f'Overlap {int(width)}')
                for width in sorted(set(edge_widths), reverse=True)
            ]

            plt.legend(
                handles=legend_elements + edge_legend_elements,
                loc='upper left',
                bbox_to_anchor=(1, 1),
                fontsize=8,
                title="Legend"
            )

            # Add a title
            plt.title(f"{name_of_topics} Overlap Graph", fontsize=14)

            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            graph_filename = f"topic_overlap_graph_{timestamp}.png"  # Include timestamp in the filename
            graph_path = os.path.join(output_subfolder, graph_filename)
            
            # Save the graph
            plt.savefig(graph_path, dpi=300, bbox_inches='tight')
            print(f"Topic overlap graph saved at {graph_path}")

        except Exception as e:
            print(f"Error generating topic overlap graph: {e}")
        finally:
            plt.close() '''