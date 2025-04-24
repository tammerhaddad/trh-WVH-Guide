"""
wvh_guide.run
-------------

High-level entrypoint for the WVH building navigation guide.

This module locates the bundled map data, computes a path between two nodes
using Dijkstra’s algorithm, summarizes the raw directions via a Gemini LLM,
and (optionally) visualizes the path on a floorplan.
"""

import importlib.resources as pkg_resources
from typing import List, Tuple

from .llm_navigator import Navigator
from .llm_service    import Summarizer
from .visualizer     import MapVisualizer


def run(
    api_key: str,
    model: str,
    start: str,
    goal: str,
    show_map: bool = True
) -> Tuple[List[str], List[str]]:
    """Compute and summarize a navigation path through the WVH building.

    This function will automatically load the bundled `WVH.json` map file,
    find the shortest path from `start` to `goal`, ask a Gemini-based LLM
    to turn the raw step list into a human-friendly summary, and—
    if `show_map=True`—pop up a matplotlib window with the highlighted route.

    Args:
        api_key (str): Your Gemini API key.
        model (str): Gemini model name (e.g. `"gemini-2.5-pro-exp-03-25"`).
        start (str): Node ID of the starting location (e.g. `"f1_p01"`).
        goal (str): Node ID or node type (e.g. `"exit"`, `"elevator"`).
        show_map (bool, optional): If True, display a map with the path. Defaults to True.

    Returns:
        tuple[list[str], list[str]]:
            - **summary**: List of human-readable direction sentences.
            - **path**: Ordered list of node IDs from `start` to `goal`.

    Raises:
        ValueError: If no valid path is found between `start` and `goal`.
    """
    # Locate the bundled JSON map file inside this package
    map_path = pkg_resources.files("wvh_guide") / "data" / "WVH.json"

    # 1) Load map and compute raw, unformatted steps + node path
    navigator = Navigator(str(map_path))
    raw_steps, path = navigator.get_directions(start, goal)

    # 2) Generate a concise, human-friendly summary via the Summarizer LLM
    summary, latency = Summarizer(api_key, model).generate(raw_steps)

    # 3) Optionally visualize the path on a floorplan
    if show_map:
        visualizer = MapVisualizer(navigator.graph)
        visualizer.highlight_path(path)
        visualizer.show()

    return summary, path
