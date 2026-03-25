import math
import config
from neural_network import NeuralNetwork
from sensors import SensorSystem

class Car:
    _sensors = SensorSystem()

    def __init__(self, track, brain: NeuralNetwork | None = None):
        self.track = track
        self.x, self.y = track.start_pos
        self.angle     = track.start_heading
        self.speed     = 0.0
        self.alive     = True
        self.alive_time = 0
        self.fitness            = 0.0
        self.checkpoints_passed = 0
        self.next_cp_idx        = 1      
        self._last_cp_frame     = 0
        self._dist_to_next_cp   = 1000.0
        self.history: list[tuple[float, float]] = []
        self.death_reason = ""
        self.brain = brain if brain is not None else NeuralNetwork()
        self.sensor_readings:  list[float]       = [1.0] * config.NUM_SENSORS
        self.sensor_endpoints: list[tuple]       = [(0.0, 0.0)] * config.NUM_SENSORS
        self.out_steer    = 0.5
        self.out_throttle = 0.0 
        self.out_brake    = 0.0

    def update(self):
        if not self.alive:
            return
        self.alive_time += 1
        self.sensor_readings, self.sensor_endpoints = self._sensors.cast(
            self.x, self.y, self.angle, self.track.wall_segments
        )
        norm_speed = self.speed / config.MAX_SPEED
        mx, my = self.track.checkpoint_midpoint(self.next_cp_idx)
        target_angle = math.atan2(my - self.y, mx - self.x)
        angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
        norm_angle = (angle_diff / math.pi) * 0.5 + 0.5
        brain_inputs = self.sensor_readings + [norm_speed, norm_angle]
        outputs = self.brain.forward(brain_inputs)
        self.out_steer    = outputs[0]
        self.out_throttle = outputs[1]
        self.out_brake    = outputs[2]
        steering = (self.out_steer - 0.5) * 2.0
        self.angle += steering * config.MAX_STEER
        accel_force = self.out_throttle * config.ACCELERATION
        brake_force = self.out_brake * config.BRAKE_DECEL
        self.speed += accel_force - brake_force
        self.speed *= config.FRICTION
        self.speed = max(0.0, min(config.MAX_SPEED, self.speed))
        dx = math.cos(self.angle) * self.speed
        dy = math.sin(self.angle) * self.speed
        self.x += dx
        self.y += dy
        if self.alive_time > 30:
            if not self.track.is_on_track(self.x, self.y):
                self.alive = False
                self.death_reason = "CRASHED"
                self.fitness -= 200.0
                return
        self._check_checkpoint()
        cp_score = self.checkpoints_passed * 1000.0
        dist_bonus = max(0, 500 - self._dist_to_next_cp) * 0.5
        speed_bonus = self.speed * 0.2
        self.fitness = cp_score + (self.alive_time * 0.1) + dist_bonus + speed_bonus
        if (self.alive_time - self._last_cp_frame) > config.NO_PROGRESS_TIMEOUT:
            self.alive = False
            self.death_reason = "STALLED"
        if self.alive_time % 5 == 0:
            self.history.append((self.x, self.y))
            if len(self.history) > 60: self.history.pop(0)

    def _check_checkpoint(self):
        mx, my = self.track.checkpoint_midpoint(self.next_cp_idx)
        self._dist_to_next_cp = math.hypot(self.x - mx, self.y - my)
        if self._dist_to_next_cp < 80:
            self.checkpoints_passed += 1
            self.next_cp_idx        += 1
            self._last_cp_frame      = self.alive_time

    @property
    def steering_mapped(self) -> float:
        return (self.out_steer - 0.5) * 2.0

    @property
    def accel_mapped(self) -> float:
        return self.out_throttle - self.out_brake

    def corners(self):
        from config import CAR_LENGTH, CAR_WIDTH_PX
        hw, hl = CAR_WIDTH_PX / 2, CAR_LENGTH / 2
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        pts = []
        for fx, fy in ((hl, -hw), (hl, hw), (-hl, hw), (-hl, -hw)):
            pts.append((self.x + ca * fx - sa * fy, self.y + sa * fx + ca * fy))
        return pts
