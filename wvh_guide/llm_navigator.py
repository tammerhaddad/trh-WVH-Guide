"""
wvh_guide.llm_navigator
-----------------------

Provides path-finding and human-readable direction generation
for the WVH building map.
"""

import heapq
import json
from typing import Any, Dict, List, Tuple

import numpy as np


class Navigator:
    """
    Navigation system for WVH building.

    Uses Dijkstra’s algorithm to compute shortest paths on a graph
    loaded from a JSON map, and converts those paths into step-by-step
    directions suitable for human consumption.
    """

    def __init__(self, map_file: str) -> None:
        """
        Initialize the Navigator by loading the building map.

        Args:
            map_file (str): Path to the JSON file containing the
                building graph. The JSON should map node IDs (str)
                to dictionaries with keys:
                  - 'neighbors': List[str]
                  - 'x', 'y': float coordinates
                  - 'floor': int floor number
                  - optional 'type': str e.g. 'exit', 'elevator'
        
        Raises:
            FileNotFoundError: If `map_file` does not exist.
            json.JSONDecodeError: If the file is not valid JSON.
        """
        with open(map_file, 'r') as f:
            self.graph: Dict[str, Dict[str, Any]] = json.load(f)

    def find_path(self, start: str, goal: str) -> List[str]:
        """
        Compute the shortest path from `start` to `goal` using Dijkstra’s algorithm.

        If `goal` is not a node ID in the graph, it is treated as a node type
        (e.g. 'exit', 'elevator') and the nearest matching node is selected.

        Args:
            start (str): Node ID for the starting point.
            goal (str): Node ID or node type for the destination.

        Returns:
            List[str]: Ordered list of node IDs from `start` to `goal`.

        Raises:
            ValueError: If no path exists between `start` and `goal`.
        """
        # If goal is a type rather than a specific node, find the closest one
        if goal not in self.graph:
            goal = self._find_nearest_type(start, goal)

        # Min-heap of (accumulated_cost, node_id)
        queue: List[Tuple[float, str]] = [(0.0, start)]
        # Maps node_id -> (predecessor_node, best_cost_so_far)
        visited: Dict[str, Tuple[Any, float]] = {start: (None, 0.0)}

        while queue:
            cost, current = heapq.heappop(queue)
            if current == goal:
                # Reconstruct path by walking predecessors
                path: List[str] = []
                node: Any = current
                while node:
                    path.append(node)
                    node = visited[node][0]
                return path[::-1]

            for neighbor in self.graph[current]['neighbors']:
                # Base cost is Euclidean distance between nodes
                base_cost = self._get_distance(current, neighbor)
                # Add penalty for switching floors (except via elevator)
                if self.graph[current]['floor'] != self.graph[neighbor]['floor']:
                    if "elevator" in neighbor:
                        base_cost += 0.0
                    else:
                        base_cost += 10.0

                new_cost = cost + base_cost
                prev = visited.get(neighbor)
                if prev is None or new_cost < prev[1]:
                    visited[neighbor] = (current, new_cost)
                    heapq.heappush(queue, (new_cost, neighbor))

        raise ValueError(f"No path found from {start} to {goal}")

    def get_directions(self, start: str, goal: str) -> Tuple[List[str], List[str]]:
        """
        Generate human-readable directions from `start` to `goal`.

        Uses `find_path` to get the raw node sequence, then
        converts straight segments, turns, and floor changes
        into textual instructions.

        Args:
            start (str): Node ID of the starting point.
            goal (str): Node ID or node type of the destination.

        Returns:
            Tuple[List[str], List[str]]:
              - directions: List of instruction strings.
              - path: Full list of node IDs traveled.

        Raises:
            ValueError: If no valid path can be found.
        """
        path = self.find_path(start, goal)
        directions: List[str] = []

        # Nothing to do if path is trivial
        if len(path) < 2:
            return directions, path

        # Compute initial direction vector
        current_direction = np.array([
            self.graph[path[1]]['x'] - self.graph[path[0]]['x'],
            self.graph[path[1]]['y'] - self.graph[path[0]]['y']
        ])
        straight_start = path[0]
        i = 0

        while i < len(path) - 1:
            curr, nxt = path[i], path[i + 1]
            new_dir = np.array([
                self.graph[nxt]['x'] - self.graph[curr]['x'],
                self.graph[nxt]['y'] - self.graph[curr]['y']
            ])

            # Handle floor changes
            if self.graph[curr]['floor'] != self.graph[nxt]['floor']:
                if straight_start != curr:
                    directions.append(
                        f"From {self._get_landmark(straight_start)}, "
                        f"go straight to {self._get_landmark(curr) or curr}"
                    )
                transport = "elevator" if "elevator" in nxt else "stairs"
                directions.append(
                    f"At {curr}, take the {transport} to floor "
                    f"{self.graph[nxt]['floor']}"
                )
                straight_start = nxt
                i += 1
                # Reset direction after floor change
                if i < len(path) - 1:
                    curr, nxt = path[i], path[i + 1]
                    current_direction = np.array([
                        self.graph[nxt]['x'] - self.graph[curr]['x'],
                        self.graph[nxt]['y'] - self.graph[curr]['y']
                    ])
                continue

            # Compute turn angle between segments
            angle = np.degrees(np.arctan2(
                np.cross(current_direction, new_dir),
                np.dot(current_direction, new_dir)
            ))
            # Significant turn if > 20°
            if abs(angle) > 20:
                if straight_start != curr:
                    directions.append(
                        f"From {self._get_landmark(straight_start)}, "
                        f"go straight to {self._get_landmark(curr) or curr}"
                    )
                loc = self._get_landmark(curr) or curr
                turn = "turn left" if angle > 0 else "turn right"
                # If next step is the last one, combine turn + straight
                if i + 1 == len(path) - 1:
                    directions.append(
                        f"At {loc}, {turn} and go straight to "
                        f"{self._get_landmark(path[-1]) or path[-1]}"
                    )
                    straight_start = path[-1]
                    i += 1
                else:
                    directions.append(f"At {loc}, {turn}")
                    straight_start = curr
                    current_direction = new_dir
                    i += 1
            else:
                current_direction = new_dir
                i += 1

        # Final straight segment if any
        if straight_start != path[-1]:
            directions.append(
                f"From {self._get_landmark(straight_start)}, "
                f"go straight to {self._get_landmark(path[-1]) or path[-1]}"
            )

        return directions, path

    def _get_distance(self, n1: str, n2: str) -> float:
        """
        Compute Euclidean distance between two nodes.

        Args:
            n1 (str): Node ID of first point.
            n2 (str): Node ID of second point.

        Returns:
            float: Straight-line distance.
        """
        p1 = np.array([self.graph[n1]['x'], self.graph[n1]['y']])
        p2 = np.array([self.graph[n2]['x'], self.graph[n2]['y']])
        return float(np.linalg.norm(p2 - p1))

    def _get_landmark(self, node_id: str) -> str:
        """
        Map a node ID to a human-readable landmark description.

        Args:
            node_id (str): Node identifier.

        Returns:
            str: Landmark name (e.g. 'the elevator', 'room 101'),
                or empty string for generic corridors.
        """
        node = self.graph[node_id]
        t = node.get('type', '')
        if t == 'exit':
            return 'the building exit'
        if t == 'elevator':
            return 'the elevator'
        if t == 'stairs':
            return 'the stairwell'
        if t == 'intersection':
            return 'the intersection'
        if t == 'elevator front':
            return 'the front of the elevator'
        if t.startswith('room '):
            return t
        # Fallback: derive room from node_id suffix if numeric
        suffix = node_id[3:]
        return f'room {suffix}' if suffix.isdigit() else ''

    def _find_nearest_type(self, start: str, goal_type: str) -> str:
        """
        Locate the nearest node of a given type.

        Args:
            start (str): Node ID to start search from.
            goal_type (str): Desired node type (e.g. 'exit').

        Returns:
            str: Node ID of the nearest matching node.

        Raises:
            ValueError: If no candidates of `goal_type` exist
                or none are reachable from `start`.
        """
        candidates = [
            nid for nid, data in self.graph.items()
            if data.get('type') == goal_type
        ]
        if not candidates:
            raise ValueError(f"No locations of type '{goal_type}'")

        best_node: str = None  # type: ignore
        best_dist: float = float('inf')

        for candidate in candidates:
            try:
                path = self.find_path(start, candidate)
                total_dist = sum(
                    self._get_distance(path[i], path[i + 1])
                    for i in range(len(path) - 1)
                )
                if total_dist < best_dist:
                    best_node, best_dist = candidate, total_dist
            except ValueError:
                # Candidate not reachable; skip
                continue

        if best_node is None:
            raise ValueError(f"No reachable '{goal_type}' from {start}")

        return best_node
