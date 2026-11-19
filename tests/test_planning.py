"""路径规划单元测试（对应方案 C6）。

运行：
    cd handoff/code
    python -m unittest discover -s tests -v
"""

import os
import random
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.planning import astar, patrol_order, plan  # noqa: E402

CELL = 0.5


def open_grid(cols=12, rows=10):
    return [[0] * cols for _ in range(rows)]


def path_is_free(grid, path):
    for x, y in path:
        r, c = int(y // CELL), int(x // CELL)
        if grid[r][c] != 0:
            return False
    return True


class TestAStar(unittest.TestCase):
    def test_straight_line(self):
        grid = open_grid()
        res = plan(grid, (0.25, 0.25), (4.75, 0.25))
        self.assertTrue(res["reachable"])
        self.assertAlmostEqual(res["cost"], 4.5, delta=0.01)
        self.assertTrue(path_is_free(grid, res["path"]))

    def test_goes_around_wall_through_gap(self):
        grid = open_grid(cols=15, rows=12)
        for r in range(12):
            grid[r][7] = 1
        grid[6][7] = 0  # 留一个缺口
        res = plan(grid, (1.0, 1.0), (5.5, 4.0))
        self.assertTrue(res["reachable"])
        self.assertTrue(path_is_free(grid, res["path"]))
        self.assertTrue(any(int(x // CELL) == 7 and int(y // CELL) == 6 for x, y in res["path"]))

    def test_unreachable_returns_no_path(self):
        grid = open_grid(cols=20, rows=10)
        for r in range(10):
            grid[r][10] = 1  # 完全封死
        res = plan(grid, (1.0, 1.0), (8.0, 1.0))
        self.assertFalse(res["reachable"])
        self.assertEqual(res["code"], "NO_PATH")
        self.assertEqual(res["path"], [])

    def test_out_of_bounds(self):
        res = plan(open_grid(), (1.0, 1.0), (-1.0, 1.0))
        self.assertFalse(res["reachable"])
        self.assertEqual(res["code"], "OUT_OF_BOUNDS")

    def test_goal_in_obstacle(self):
        grid = open_grid()
        grid[5][5] = 1
        res = plan(grid, (1.0, 1.0), (2.75, 2.75))
        self.assertFalse(res["reachable"])
        self.assertEqual(res["code"], "OBSTACLE_BLOCKED")

    def test_no_corner_cutting(self):
        grid = [[0, 1], [1, 0]]
        cells, cost = astar(grid, (0, 0), (1, 1))
        self.assertIsNone(cells)

    def test_performance(self):
        random.seed(7)
        grid = [[1 if random.random() < 0.12 else 0 for _ in range(40)] for _ in range(30)]
        grid[1][1] = 0
        grid[28][38] = 0
        started = time.perf_counter()
        for _ in range(20):
            plan(grid, (0.75, 0.75), (19.25, 14.25))
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed / 20, 0.05)


class TestPatrolOrder(unittest.TestCase):
    def test_visits_all_points(self):
        pts = [(5.0, 5.0), (1.0, 1.0), (9.0, 1.0), (9.0, 9.0)]
        order = patrol_order(pts, (0.0, 0.0))
        self.assertEqual(sorted(order), [0, 1, 2, 3])
        self.assertEqual(order[0], 1)  # 离起点最近

    def test_nearest_neighbor_beats_naive(self):
        pts = [(9.0, 9.0), (1.0, 1.0), (9.0, 1.0), (1.0, 9.0)]

        def length(order):
            cur, total = (0.0, 0.0), 0.0
            for i in order:
                total += ((cur[0] - pts[i][0]) ** 2 + (cur[1] - pts[i][1]) ** 2) ** 0.5
                cur = pts[i]
            return total

        self.assertLess(length(patrol_order(pts, (0.0, 0.0))), length([0, 1, 2, 3]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
