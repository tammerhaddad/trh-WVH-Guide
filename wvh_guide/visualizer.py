"""
wvh_guide.visualizer
--------------------

Provides functionality to visualize the WVH building floorplans
and highlight navigation paths on a 2×2 grid of matplotlib subplots.
"""

from typing import Dict, List, Optional

import networkx as nx
import matplotlib.pyplot as plt


class MapVisualizer:
    """
    Visualization class for WVH guide system.

    Draws each building floor on its own subplot, and can highlight
    a computed path in green. Uses NetworkX for graph layout and
    matplotlib for rendering.
    """

    def __init__(self, graph: Dict[str, Dict]) -> None:
        """
        Initialize the visualizer with the building graph.

        Args:
            graph (Dict[str, Dict]): Mapping of node IDs to their
                attributes, including 'x', 'y', 'floor', and 'neighbors'.

        Attributes:
            subplots (List[nx.Graph]): One NetworkX graph per floor.
            fig (plt.Figure): The matplotlib Figure containing 4 axes.
            axes (List[plt.Axes]): Flattened list of the 4 subplot Axes.
            current_path (Optional[List[str]]): Node sequence to highlight.
        """
        self.graph: Dict[str, Dict] = graph
        # Create an empty NetworkX Graph for each of 4 floors
        self.subplots: List[nx.Graph] = [nx.Graph() for _ in range(4)]
        # Set up a 2×2 grid of subplots
        self.fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
        self.axes: List[plt.Axes] = axes.flatten()
        self.current_path: Optional[List[str]] = None

    def setup_display(self) -> None:
        """
        Draw all four floors with nodes and edges.

        Uses a fixed color palette per floor and arranges titles
        and margins before tightening the overall layout.
        """
        colors = ['purple', 'orange', 'red', 'lightblue']
        for floor in range(1, 5):
            ax = self.axes[floor - 1]
            color = colors[floor - 1]
            self._draw_floor(floor, color, ax)
            ax.set_title(f'Floor {floor}', fontsize=14, pad=20)
            ax.margins(0.1)
        plt.tight_layout(pad=3.0)

    def _draw_floor(self, floor: int, color: str, ax: plt.Axes) -> None:
        """
        Populate a single floor’s subplot with nodes and dashed edges.

        Args:
            floor (int): Floor number (1–4).
            color (str): Base color for nodes and edges on this floor.
            ax (plt.Axes): Axis to draw onto.
        """
        G = self.subplots[floor - 1]

        # Add nodes for this floor with positions
        for node_id, data in self.graph.items():
            if data['floor'] == floor:
                G.add_node(node_id, pos=(data['x'], data['y']))

        # Collect edges for this floor
        edges = []
        for node_id, data in self.graph.items():
            if data['floor'] == floor:
                for nbr in data['neighbors']:
                    if nbr.startswith(f'f{floor}'):
                        G.add_edge(node_id, nbr)
                        if (node_id, nbr) not in edges:
                            edges.append((node_id, nbr))

        pos = nx.get_node_attributes(G, 'pos')
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            edge_color=color,
            style='dashed',
            alpha=0.6,
            ax=ax
        )

        # Determine node sizes and labels
        sizes: List[int] = []
        labels: Dict[str, str] = {}
        for node in G.nodes():
            if self.current_path and node in self.current_path:
                sizes.append(80)
                # Label only the node number for clarity
                labels[node] = node.split('_')[-1]
            elif any(key in node.lower() for key in ['elevator', 'stairs']):
                sizes.append(60)
                labels[node] = 'Elevator' if 'elevator' in node.lower() else 'Stairs'
            else:
                sizes.append(30)

        nx.draw_networkx_nodes(
            G, pos,
            node_size=sizes,
            node_color=color,
            alpha=0.8,
            ax=ax
        )
        if labels:
            nx.draw_networkx_labels(
                G, pos, labels,
                font_size=8,
                font_weight='bold',
                bbox=dict(
                    facecolor='white',
                    edgecolor='gray',
                    alpha=0.95,
                    pad=3,
                    boxstyle='round,pad=0.5'
                ),
                ax=ax
            )

    def highlight_path(self, path: List[str]) -> None:
        """
        Highlight a path on the existing floorplan display.

        Clears and redraws the base display, then overlays the path
        as solid green edges of width 2.

        Args:
            path (List[str]): Ordered list of node IDs to highlight.
        """
        self.current_path = path
        # Clear each axis before re-drawing
        for ax in self.axes:
            ax.clear()
        self.setup_display()

        # Draw each consecutive edge in green
        for u, v in zip(path[:-1], path[1:]):
            floor = int(u[1])  # assumes format 'f{floor}_...'
            G = self.subplots[floor - 1]
            pos = nx.get_node_attributes(G, 'pos')
            # Ensure nodes exist in G
            if not G.has_edge(u, v):
                G.add_node(u, pos=(self.graph[u]['x'], self.graph[u]['y']))
                G.add_node(v, pos=(self.graph[v]['x'], self.graph[v]['y']))
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                edge_color='green',
                width=2,
                ax=self.axes[floor - 1]
            )
        plt.draw()

    def show(self, block: bool = True) -> None:
        """
        Display the matplotlib figure.

        Args:
            block (bool, optional): Whether to block execution until
                the window is closed. Defaults to True.
        """
        plt.show(block=block)
