import math
from config import MAX_SENSOR_RANGE, SENSOR_ANGLES_DEG, NUM_SENSORS

def _ray_segment_t(ox, oy, dx, dy, ax, ay, bx, by):
    sx = bx - ax
    sy = by - ay
    denom = dx * (-sy) - dy * (-sx)
    if abs(denom) < 1e-10:
        return None
    rx = ax - ox
    ry = ay - oy
    t = (rx * (-sy) - ry * (-sx)) / denom
    u = (dx * ry - dy * rx) / denom
    if t < 1e-3 or u < 0.0 or u > 1.0:
        return None
    return t

class SensorSystem:
    _angles_rad = [math.radians(a) for a in SENSOR_ANGLES_DEG]

    def cast(self, cx: float, cy: float, heading: float, wall_segments):
        readings  = []
        endpoints = []
        for offset in self._angles_rad:
            angle = heading + offset
            dx, dy = math.cos(angle), math.sin(angle)
            min_t  = MAX_SENSOR_RANGE
            for (ax, ay), (bx, by) in wall_segments:
                t = _ray_segment_t(cx, cy, dx, dy, ax, ay, bx, by)
                if t is not None and t < min_t:
                    min_t = t
            readings.append(min_t / MAX_SENSOR_RANGE)
            endpoints.append((cx + dx * min_t, cy + dy * min_t))
        return readings, endpoints
