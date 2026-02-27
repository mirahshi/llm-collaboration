"""
Maze task for LLM training, inspired by the collaborative maze-solving benchmark
from arxiv.org/abs/2511.02687.

Each sample is a flat string:  maze=solution_path
Two masked views (m1, m2) are generated so that m1 ∪ m2 = full maze.

Symbols:  @ start  * goal  . path  # wall  ? hidden
"""

from __future__ import annotations

import random
from collections import deque
from typing import Iterator


# ---------------------------------------------------------------------------
# Maze generation
# ---------------------------------------------------------------------------

def generate_maze(
    width: int,
    height: int,
    wall_density: float = 0.30,
    seed: int | None = None,
) -> list[list[str]]:
    """Return a height×width grid guaranteed to have at least one solution.

    Start = top-left (0,0), goal = bottom-right (height-1, width-1).
    """
    rng = random.Random(seed)

    while True:
        grid = [["." for _ in range(width)] for _ in range(height)]
        for r in range(height):
            for c in range(width):
                if (r, c) == (0, 0) or (r, c) == (height - 1, width - 1):
                    continue
                if rng.random() < wall_density:
                    grid[r][c] = "#"
        grid[0][0] = "@"
        grid[height - 1][width - 1] = "*"

        # ensure solvable
        if _bfs_single(grid, width, height) is not None:
            return grid


# ---------------------------------------------------------------------------
# Pathfinding
# ---------------------------------------------------------------------------

_DIRS = [(-1, 0, "up"), (1, 0, "down"), (0, -1, "left"), (0, 1, "right")]


def _bfs_single(
    grid: list[list[str]], width: int, height: int
) -> list[tuple[int, int]] | None:
    """Return one shortest path (list of (row, col)) or None."""
    start, goal = (0, 0), (height - 1, width - 1)
    queue: deque[list[tuple[int, int]]] = deque([[start]])
    visited = {start}

    while queue:
        path = queue.popleft()
        r, c = path[-1]
        if (r, c) == goal:
            return path
        for dr, dc, _ in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in visited and grid[nr][nc] != "#":
                visited.add((nr, nc))
                queue.append(path + [(nr, nc)])
    return None


def _bfs_all(
    grid: list[list[str]], width: int, height: int
) -> list[list[tuple[int, int]]]:
    """Return *all* shortest paths."""
    start, goal = (0, 0), (height - 1, width - 1)
    queue: deque[list[tuple[int, int]]] = deque([[start]])
    # track min cost to reach each cell
    best_cost: dict[tuple[int, int], int] = {start: 0}
    results: list[list[tuple[int, int]]] = []
    best_len: int | None = None

    while queue:
        path = queue.popleft()
        if best_len is not None and len(path) > best_len:
            break
        r, c = path[-1]
        if (r, c) == goal:
            best_len = len(path)
            results.append(path)
            continue
        for dr, dc, _ in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr][nc] != "#":
                new_cost = len(path)
                if (nr, nc) not in best_cost or new_cost <= best_cost[(nr, nc)]:
                    best_cost[(nr, nc)] = new_cost
                    queue.append(path + [(nr, nc)])
    return results


def find_paths(
    grid: list[list[str]], width: int, height: int, *, all_paths: bool = False
) -> list[list[tuple[int, int]]]:
    """Return either a single shortest path or all shortest paths."""
    if all_paths:
        return _bfs_all(grid, width, height)
    p = _bfs_single(grid, width, height)
    return [p] if p is not None else []


# ---------------------------------------------------------------------------
# Path → direction string
# ---------------------------------------------------------------------------

_DIR_NAMES = {(-1, 0): "u", (1, 0): "d", (0, -1): "l", (0, 1): "r"}


def path_to_directions(path: list[tuple[int, int]]) -> str:
    """Convert coordinate path to compact direction string like 'ddrrd'."""
    return "".join(
        _DIR_NAMES[(r2 - r1, c2 - c1)]
        for (r1, c1), (r2, c2) in zip(path, path[1:])
    )


def path_to_coords(path: list[tuple[int, int]]) -> str:
    """Convert coordinate path to semicolon-separated (row,col) string."""
    return ";".join(f"({r},{c})" for r, c in path)


# ---------------------------------------------------------------------------
# Masking – create two complementary partial views
# ---------------------------------------------------------------------------

def mask_maze(
    grid: list[list[str]],
    width: int,
    height: int,
    seed: int | None = None,
) -> tuple[list[list[str]], list[list[str]]]:
    """Split *grid* into two views where hidden cells are replaced by '?'.

    Start (@) and goal (*) are always visible in both views.
    Every other cell is randomly assigned to exactly one view; the other view
    sees '?' for that cell.
    """
    rng = random.Random(seed)
    m1 = [["?" for _ in range(width)] for _ in range(height)]
    m2 = [["?" for _ in range(width)] for _ in range(height)]

    for r in range(height):
        for c in range(width):
            ch = grid[r][c]
            if ch in ("@", "*"):
                m1[r][c] = ch
                m2[r][c] = ch
            else:
                if rng.random() < 0.5:
                    m1[r][c] = ch
                else:
                    m2[r][c] = ch
    return m1, m2


# ---------------------------------------------------------------------------
# Flat-string encoding
# ---------------------------------------------------------------------------

def grid_to_flat(grid: list[list[str]]) -> str:
    """Row-major flat string (no separators between cells, newline between rows)."""
    return "\n".join("".join(row) for row in grid)


def flat_to_grid(flat: str) -> list[list[str]]:
    """Inverse of grid_to_flat."""
    return [list(row) for row in flat.split("\n")]


def format_sample(maze_flat: str, solution: str) -> str:
    """Produce the LLM training string as a single line:  maze=solution_path"""
    return f"{maze_flat.replace(chr(10), '')}={solution}"


# ---------------------------------------------------------------------------
# High-level generator
# ---------------------------------------------------------------------------

def generate_samples(
    width: int = 6,
    height: int = 6,
    wall_density: float = 0.30,
    n: int = 1,
    all_paths: bool = False,
    path_format: str = "directions",
    seed: int | None = None,
    mask: bool = True
) -> Iterator[dict]:
    """Yield dicts with keys: maze, m1, m2, paths, samples.

    Parameters
    ----------
    width, height : maze dimensions
    wall_density  : fraction of interior cells that are walls
    n             : number of mazes to generate
    all_paths     : if True, enumerate all shortest paths
    path_format   : "directions" or "coords"
    seed          : reproducibility seed
    """
    rng_seed = random.Random(seed)
    fmt = path_to_directions if path_format == "directions" else path_to_coords

    for _ in range(n):
        s = rng_seed.randint(0, 2**31)
        grid = generate_maze(width, height, wall_density, seed=s)
        paths = find_paths(grid, width, height, all_paths=all_paths)
        if mask:
            m1, m2 = mask_maze(grid, width, height, seed=s + 1)
        else:
            m1, m2 = grid, grid

        maze_flat = grid_to_flat(grid)
        m1_flat = grid_to_flat(m1)
        m2_flat = grid_to_flat(m2)

        solutions = [fmt(p) for p in paths]
        samples_full = [format_sample(maze_flat, sol) for sol in solutions]
        samples_m1 = [format_sample(m1_flat, sol) for sol in solutions]
        samples_m2 = [format_sample(m2_flat, sol) for sol in solutions]

        yield {
            "grid": grid,
            "maze": maze_flat,
            "m1": m1_flat,
            "m2": m2_flat,
            "paths": paths,
            "solutions": solutions,
            "samples_full": samples_full,
            "samples_m1": samples_m1,
            "samples_m2": samples_m2,
        }


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def visualize_grid(flat: str, label: str = "") -> str:
    """Pretty-print a flat maze string back as a grid with border."""
    grid = flat_to_grid(flat)
    height, width = len(grid), len(grid[0])
    lines: list[str] = []
    if label:
        lines.append(f"  {label}")
    lines.append("  +" + "-" * (width * 2 - 1) + "+")
    for row in grid:
        cells = " ".join(row)
        lines.append(f"  |{cells}|")
    lines.append("  +" + "-" * (width * 2 - 1) + "+")
    return "\n".join(lines)


def visualize_path_on_grid(grid: list[list[str]], path: list[tuple[int, int]]) -> str:
    """Return a grid string with the solution path marked as 'o'."""
    import copy
    g = copy.deepcopy(grid)
    for r, c in path[1:-1]:  # keep @ and * visible
        g[r][c] = "o"
    return grid_to_flat(g)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Maze task generator for LLM training")
    parser.add_argument("--width", type=int, default=6)
    parser.add_argument("--height", type=int, default=6)
    parser.add_argument("--wall-density", type=float, default=0.30)
    parser.add_argument("--n", type=int, default=3, help="number of mazes")
    parser.add_argument("--all-paths", action="store_true", help="enumerate all shortest paths")
    parser.add_argument("--path-format", choices=["directions", "coords"], default="directions")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for i, data in enumerate(generate_samples(
        width=args.width,
        height=args.height,
        wall_density=args.wall_density,
        n=args.n,
        all_paths=args.all_paths,
        path_format=args.path_format,
        seed=args.seed,
    )):
        print(f"\n{'='*60}")
        print(f"  MAZE {i+1}")
        print(f"{'='*60}")

        # Full maze
        print(visualize_grid(data["maze"], label="Full maze"))

        # Solution path overlaid
        for j, path in enumerate(data["paths"]):
            solved = visualize_path_on_grid(data["grid"], path)
            label = f"Solution {j+1}" if len(data["paths"]) > 1 else "Solution"
            print(visualize_grid(solved, label=label))

        # Masked views
        print(visualize_grid(data["m1"], label="Agent 1 view (m1)"))
        print(visualize_grid(data["m2"], label="Agent 2 view (m2)"))

        # Flat training strings
        print("\n  Training samples (full maze):")
        for s in data["samples_full"]:
            print(f"    {s}")
        print("\n  Training samples (m1 - agent 1):")
        for s in data["samples_m1"]:
            print(f"    {s}")
        print("\n  Training samples (m2 - agent 2):")
        for s in data["samples_m2"]:
            print(f"    {s}")
        print()
