import math
import random
from config import NUM_CHECKPOINTS, OUTER_RX

def _seg_dist_sq(px, py, x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0:
        return (px - x1)**2 + (py - y1)**2
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t))
    return (px - (x1 + t * dx))**2 + (py - (y1 + t * dy))**2

class Track:
    def __init__(self):
        self.track_width   = 120.0
        self.center_pts    = []
        self.outer_pts     = []
        self.inner_pts     = []
        self.wall_segments = []
        self.checkpoints   = []
        self.start_pos     = (0.0, 0.0)
        self.start_heading = 0.0
        self._build_track()

    def _smooth(self, pts, iterations=3):
        for _ in range(iterations):
            new_pts = []
            for n in range(len(pts)):
                p1 = pts[n]
                p2 = pts[(n + 1) % len(pts)]
                new_pts.append((0.75 * p1[0] + 0.25 * p2[0], 0.75 * p1[1] + 0.25 * p2[1]))
                new_pts.append((0.25 * p1[0] + 0.75 * p2[0], 0.25 * p1[1] + 0.75 * p2[1]))
            pts = new_pts
        return pts

    def _build_track(self):
        base = []
        n_base = 15
        for i in range(n_base):
            a = (2 * math.pi / n_base) * i
            a += (random.random() - 0.5) * (math.pi / n_base) * 0.7
            r = OUTER_RX * (0.4 + random.random() * 0.9)
            base.append((math.cos(a) * r, math.sin(a) * r))

        self.center_pts = self._smooth(base, iterations=4)
        n = len(self.center_pts)
        self.outer_pts = []
        self.inner_pts = []
        hw = self.track_width / 2.0

        for i in range(n):
            p1 = self.center_pts[i-1]
            p3 = self.center_pts[(i+1)%n]
            dx, dy = p3[0] - p1[0], p3[1] - p1[1]
            mag    = math.hypot(dx, dy)
            if mag < 1e-6:
                normal = (1, 0)
            else:
                normal = (-dy/mag, dx/mag)
            p2 = self.center_pts[i]
            self.outer_pts.append((p2[0] + normal[0]*hw, p2[1] + normal[1]*hw))
            self.inner_pts.append((p2[0] - normal[0]*hw, p2[1] - normal[1]*hw))

        self.wall_segments = []
        for p in [self.outer_pts, self.inner_pts]:
            for i in range(len(p)):
                self.wall_segments.append((p[i], p[(i+1)%len(p)]))

        self.checkpoints = []
        step = max(1, n // NUM_CHECKPOINTS)
        for i in range(0, n, step):
            self.checkpoints.append((self.outer_pts[i], self.inner_pts[i]))

        self.start_pos = self.center_pts[0]
        p_next = self.center_pts[1]
        self.start_heading = math.atan2(p_next[1] - self.start_pos[1], p_next[0] - self.start_pos[0])

    def is_on_track(self, x, y):
        min_dist_sq = 1e10
        half_width_sq = (self.track_width / 2.0)**2
        safe_half_width_sq = (self.track_width / 2.0 + 2.0)**2

        for i in range(len(self.center_pts)):
            p1 = self.center_pts[i]
            p2 = self.center_pts[(i+1) % len(self.center_pts)]
            d2 = _seg_dist_sq(x, y, p1[0], p1[1], p2[0], p2[1])
            if d2 < min_dist_sq:
                min_dist_sq = d2
            if min_dist_sq < safe_half_width_sq:
                return True
        return min_dist_sq <= safe_half_width_sq

    def checkpoint_midpoint(self, idx):
        cp_idx = (idx * (len(self.center_pts) // NUM_CHECKPOINTS)) % len(self.center_pts)
        return self.center_pts[cp_idx]
