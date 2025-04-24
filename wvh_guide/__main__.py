#!/usr/bin/env python3
"""
wvh-guide CLI
-------------

Command-line interface for the WVH building navigation guide.

This script loads environment variables, parses user arguments,
and invokes the high-level `run()` function from the `wvh_guide` package
to compute and (optionally) visualize a path through the WVH building.
"""

import os
import argparse
from dotenv import load_dotenv

from wvh_guide import run


def main() -> None:
    """
    Entry point for the wvh-guide CLI.

    Loads API credentials from the environment, parses command-line options,
    and calls `wvh_guide.run()` to compute and summarize the navigation path.

    Command-line arguments:
        --api_key   Gemini API key (overrides $API_KEY from .env)
        --model     Gemini model identifier (e.g. "gemini-2.5-pro-exp-03-25")
        --start     Node ID of the starting location (required)
        --goal      Node ID or type of the target location (required)
        --no-map    If set, skip the matplotlib visualization
    """
    # 0) Load environment variables from a .env file (if present)
    load_dotenv()

    # 1) Set up argument parser
    parser = argparse.ArgumentParser(
        prog="wvh-guide",
        description="Compute and summarize a navigation path through WVH building."
    )
    parser.add_argument(
        "--api_key",
        default=None,
        help="Your Gemini API key (overrides $API_KEY from environment)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Gemini model name, e.g. 'gemini-2.5-pro-exp-03-25'"
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start node ID (e.g. 'f1_p01')"
    )
    parser.add_argument(
        "--goal",
        required=True,
        help="Goal node ID or type (e.g. 'exit', 'elevator')"
    )
    parser.add_argument(
        "--no-map",
        action="store_true",
        help="Skip displaying the map visualization"
    )

    args = parser.parse_args()

    # 2) Invoke the core library function
    #    Automatically loads the bundled WVH.json, computes path, summarizes it,
    #    and (unless --no-map) displays the map.
    run(
        api_key=args.api_key,
        model=args.model,
        start=args.start,
        goal=args.goal,
        show_map=not args.no_map
    )


if __name__ == "__main__":
    main()
