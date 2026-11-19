"""路径规划与巡检排序（对应方案中的 C · 算法）。

约定（与 docs/02_小组分工.md 第七章一致）：
- 栅格 grid[row][col]，0 可通行、1 障碍，row 0 在最下方，每格 cell 米。
- 8 邻域 A*，禁止“斜穿两个障碍的夹角”，启发函数用 octile。
- 对外只给世界坐标（米）的路径点列，供前端画线和仿真执行。

命令行自测：
    python core/planning.py
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple

SQRT2 = math.sqrt(2.0)
NEIGHBORS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _rows_cols(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    return len(grid), len(grid[0])


def _passable(grid: Sequence[Sequence[int]], r: int, c: int) -> bool:
    rows, cols = _rows_cols(grid)
    return 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0


def _octile(dr: int, dc: int) -> float:
    a, b = abs(dr), abs(dc)
    return (a + b) + (SQRT2 - 2.0) * min(a, b)


def astar(grid: Sequence[Sequence[int]], start: Tuple[int, int],
          goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], float]:
    """返回 (格子路径, 代价)；不可达返回 (None, inf)。"""
    if not _passable(grid, *start) or not _passable(grid, *goal):
        return None, math.inf
    if start == goal:
        return [start], 0.0

    open_heap: List[Tuple[float, float, Tuple[int, int]]] = [(_octile(goal[0] - start[0], goal[1] - start[1]), 0.0, start)]
    came: Dict[Tuple[int, int], Tuple[int, int]] = {}
    best: Dict[Tuple[int, int], float] = {start: 0.0}

    while open_heap:
        _, g, cur = heapq.heappop(open_heap)
        if cur == goal:
            path = [cur]
            while cur in came:
                cur = came[cur]
                path.append(cur)
            return list(reversed(path)), g
        if g > best.get(cur, math.inf):
            continue
        r, c = cur
        for dr, dc in NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not _passable(grid, nr, nc):
                continue
            if dr and dc and not (_passable(grid, r + dr, c) and _passable(grid, r, c + dc)):
                continue  # 禁止斜穿墙角
            ng = g + (SQRT2 if dr and dc else 1.0)
            if ng < best.get((nr, nc), math.inf):
                best[(nr, nc)] = ng
                came[(nr, nc)] = cur
                heapq.heappush(open_heap, (ng + _octile(goal[0] - nr, goal[1] - nc), ng, (nr, nc)))
    return None, math.inf


def _to_cell(x: float, y: float, cell: float, rows: int, cols: int) -> Tuple[int, int]:
    r = min(rows - 1, max(0, int(y // cell)))
    c = min(cols - 1, max(0, int(x // cell)))
    return r, c


def _to_world(r: int, c: int, cell: float) -> Tuple[float, float]:
    return round((c + 0.5) * cell, 4), round((r + 0.5) * cell, 4)


def plan(grid: Sequence[Sequence[int]], start_xy: Tuple[float, float],
         goal_xy: Tuple[float, float], cell: float = 0.5) -> Dict[str, object]:
    """对外接口：世界坐标进，世界坐标路径出。"""
    rows, cols = _rows_cols(grid)
    width, height = cols * cell, rows * cell
    sx, sy = start_xy
    gx, gy = goal_xy

    for x, y in (start_xy, goal_xy):
        if x < 0 or y < 0 or x >= width or y >= height:
            return {"path": [], "cost": None, "reachable": False,
                    "code": "OUT_OF_BOUNDS", "message": "起点或目标点超出地图范围"}

    start = _to_cell(sx, sy, cell, rows, cols)
    goal = _to_cell(gx, gy, cell, rows, cols)

    for r, c in (start, goal):
        if grid[r][c] == 1:
            return {"path": [], "cost": None, "reachable": False,
                    "code": "OBSTACLE_BLOCKED", "message": "起点或目标点在障碍物内"}

    cells, cost = astar(grid, start, goal)
    if cells is None:
        return {"path": [], "cost": None, "reachable": False,
                "code": "NO_PATH", "message": "目标点不可达，请换一个目标"}

    points = [_to_world(r, c, cell) for r, c in cells]
    points[0] = (round(sx, 4), round(sy, 4))
    points[-1] = (round(gx, 4), round(gy, 4))
    return {"path": points, "cost": round(cost * cell, 3), "reachable": True,
            "code": None, "cells": cells}


def patrol_order(points: Sequence[Tuple[float, float]],
                 start: Tuple[float, float]) -> List[int]:
    """最近邻排序：返回巡检点的访问顺序（下标）。点数少时足够用。"""
    remaining = list(range(len(points)))
    order: List[int] = []
    cur = start
    while remaining:
        nxt = min(remaining, key=lambda i: math.dist(cur, points[i]))
        order.append(nxt)
        remaining.remove(nxt)
        cur = points[nxt]
    return order


def patrol_route(points: Sequence[Tuple[float, float]],
                 start: Tuple[float, float]) -> List[Tuple[float, float]]:
    """按最近邻顺序返回巡检点列。"""
    return [points[i] for i in patrol_order(points, start)]


if __name__ == "__main__":
    # 40×30 地图，中间一堵带缺口的墙
    rows, cols = 30, 40
    grid = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        grid[r][20] = 1
    for r in range(12, 17):
        grid[r][20] = 0

    res = plan(grid, (1.0, 1.0), (18.0, 14.0))
    print("可达:", res["reachable"], "代价:", res["cost"], "点数:", len(res["path"]))
    print("前 3 个点:", res["path"][:3])

    bad = plan(grid, (1.0, 1.0), (-1.0, 1.0))
    print("越界示例:", bad["code"])
