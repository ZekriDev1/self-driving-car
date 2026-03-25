import math
import pygame
import config
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, SIM_WIDTH, UI_WIDTH,
    CAR_LENGTH, CAR_WIDTH_PX,
    C_BG, C_TRACK, C_WALL, C_UI_BG, C_UI_LINE,
    C_CAR_BEST, C_CAR_SECOND, C_CAR_NORMAL,
    C_SENSOR_LINE, C_SENSOR_HIT,
    C_TEXT, C_TEXT_DIM, C_ACCENT,
    C_NN_POS, C_NN_NEG, C_NN_NODE,
    NUM_SENSORS, NN_LAYERS, OUTER_RX
)

def _blend(col, alpha, bg=(18, 18, 28)):
    a = alpha / 255
    return tuple(int(col[i] * a + bg[i] * (1 - a)) for i in range(3))

class Camera:
    def __init__(self, lerp: float = 0.06):
        self.cx   = 0.0
        self.cy   = 0.0
        self.lerp = lerp
        self.zoom = 1.0
        self.vx   = SIM_WIDTH  // 2
        self.vy   = SCREEN_HEIGHT // 2

    def follow(self, target_x: float, target_y: float):
        self.cx += (target_x - self.cx) * self.lerp
        self.cy += (target_y - self.cy) * self.lerp

    def update_viewport(self, sim_width: int, screen_height: int):
        self.vx = sim_width // 2
        self.vy = screen_height // 2

    def world_to_screen(self, wx: float, wy: float):
        return (int(self.vx + (wx - self.cx) * self.zoom),
                int(self.vy + (wy - self.cy) * self.zoom))

    def w2s(self, pt):
        return self.world_to_screen(pt[0], pt[1])

class Renderer:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.camera = Camera()
        self.w, self.h = screen.get_size()
        self.ui_width = UI_WIDTH
        self.sim_width = max(100, self.w - self.ui_width)
        self.camera.update_viewport(self.sim_width, self.h)
        self._recreate_surfaces()
        pygame.font.init()
        self._init_fonts()

    def _init_fonts(self):
        self.font_lg  = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_md  = pygame.font.SysFont("Consolas", 16)
        self.font_sm  = pygame.font.SysFont("Consolas", 13)

    def _recreate_surfaces(self):
        self.sim_surf = pygame.Surface((self.sim_width, self.h))
        self.ui_surf  = pygame.Surface((self.ui_width,  self.h))

    def handle_resize(self, width: int, height: int):
        self.w, self.h = width, height
        self.sim_width = max(100, self.w - self.ui_width)
        self.camera.update_viewport(self.sim_width, self.h)
        self._recreate_surfaces()

    def render(self, track, cars, generation: int, best_fitness: float, target_pop: int, sim_speed: int, all_cars_pool=None):
        if all_cars_pool is None:
            all_cars_pool = cars
        alive_list = sorted([c for c in cars if c.alive], key=lambda c: c.fitness, reverse=True)
        ranked_all = sorted(all_cars_pool, key=lambda c: c.fitness, reverse=True)
        best   = ranked_all[0] if ranked_all else None
        second = ranked_all[1] if len(ranked_all) > 1 else None
        target = alive_list[0] if alive_list else best
        if target:
            self.camera.follow(target.x, target.y)
        self._draw_sim(track, cars, best, second, generation, best_fitness)
        self._draw_ui(best, second, generation, best_fitness, len(alive_list), target_pop, sim_speed)
        self.screen.blit(self.sim_surf, (0, 0))
        self.screen.blit(self.ui_surf,  (self.sim_width, 0))

    def _draw_sim(self, track, cars, best, second, generation, best_fitness):
        surf = self.sim_surf
        surf.fill(C_BG)
        self._draw_track(surf, track)
        self._draw_checkpoints(surf, track)
        if best and best.history:
            points = [self.camera.w2s(p) for p in best.history]
            if len(points) > 1:
                pygame.draw.lines(surf, _blend((0, 255, 150), 120), False, points, max(1, int(3 * self.camera.zoom)))
        best_set = {id(best), id(second)}
        for car in cars:
            if id(car) not in best_set:
                self._draw_car(surf, car, C_CAR_NORMAL, sensors=False)
        if second: self._draw_car(surf, second, C_CAR_SECOND, sensors=False)
        if best:   self._draw_car(surf, best,   C_CAR_BEST,   sensors=True)
        gen_surf = self.font_lg.render(f"GEN  {generation}", True, C_ACCENT)
        surf.blit(gen_surf, (14, self.h - 36))
        alive_count = sum(1 for c in cars if c.alive)
        a_surf = self.font_sm.render(f"Alive  {alive_count}/{len(cars)}", True, C_TEXT_DIM)
        surf.blit(a_surf, (14, self.h - 58))

    def _draw_track(self, surf, track):
        cam = self.camera
        outer_s = [cam.w2s(p) for p in track.outer_pts]
        inner_s = [cam.w2s(p) for p in track.inner_pts]
        pygame.draw.polygon(surf, C_TRACK, outer_s)
        pygame.draw.polygon(surf, C_BG,    inner_s)
        pygame.draw.polygon(surf, C_WALL,  outer_s, max(1, int(2 * cam.zoom)))
        pygame.draw.polygon(surf, C_WALL,  inner_s, max(1, int(2 * cam.zoom)))

    def _draw_checkpoints(self, surf, track):
        cam = self.camera
        cp_col = _blend((255, 230, 50), 45)
        for (op, ip) in track.checkpoints:
            pygame.draw.line(surf, cp_col, cam.w2s(op), cam.w2s(ip), max(1, int(1 * cam.zoom)))

    def _draw_car(self, surf, car, colour, sensors=False):
        cam = self.camera
        if sensors:
            self._draw_sensors(surf, car)
        final_col = (255, 50, 50) if not car.alive else colour
        corners_s = [cam.w2s(p) for p in car.corners()]
        pygame.draw.polygon(surf, final_col, corners_s)
        pygame.draw.polygon(surf, (255, 255, 255), corners_s, 1)
        fx = car.x + math.cos(car.angle) * (CAR_LENGTH / 2 + 3)
        fy = car.y + math.sin(car.angle) * (CAR_LENGTH / 2 + 3)
        pygame.draw.circle(surf, (255, 255, 255), cam.w2s((fx, fy)), max(1, int(3 * cam.zoom)))
        if not car.alive and car.death_reason:
            msg = self.font_sm.render(car.death_reason, True, (255, 100, 100))
            surf.blit(msg, cam.w2s((car.x - 25, car.y - 35)))

    def _draw_sensors(self, surf, car):
        cam = self.camera
        origin_s = cam.w2s((car.x, car.y))
        for i, ep in enumerate(car.sensor_endpoints):
            ep_s = cam.w2s(ep)
            hit  = car.sensor_readings[i] < 0.98
            col  = C_SENSOR_HIT if hit else C_SENSOR_LINE
            pygame.draw.line(surf, col, origin_s, ep_s, 1)
            pygame.draw.circle(surf, col, ep_s, max(1, int(3 * cam.zoom)))

    def _draw_ui(self, best, second, generation, global_best_fitness, alive_count, target_pop, sim_speed):
        surf = self.ui_surf
        surf.fill(C_UI_BG)
        pygame.draw.line(surf, C_UI_LINE, (0, 0), (0, self.h), 2)
        y = 18
        title = self.font_lg.render("SELF-DRIVING  CARS", True, C_ACCENT)
        surf.blit(title, (self.ui_width // 2 - title.get_width() // 2, y));  y += 36
        pygame.draw.line(surf, C_UI_LINE, (12, y), (self.ui_width - 12, y), 1);  y += 14
        y = self._stat_block(surf, y, generation, global_best_fitness, best, target_pop, sim_speed, alive_count)
        p_y = y + 10
        self._draw_text(surf, "LEARNING STATUS", (18, p_y), size=14, color=C_TEXT_DIM)
        p_y += 18
        gen_best = best.fitness if best else 0
        progress = min(1.0, gen_best / max(1, global_best_fitness)) if global_best_fitness > 0 else 0
        pygame.draw.rect(surf, (40, 40, 50), (18, p_y, self.ui_width - 36, 12), border_radius=4)
        pygame.draw.rect(surf, (0, 200, 100), (18, p_y, (self.ui_width - 36) * progress, 12), border_radius=4)
        y = p_y + 30
        pygame.draw.line(surf, C_UI_LINE, (12, y), (self.ui_width - 12, y), 1);  y += 14
        if best:
            y = self._draw_nn(surf, y, best)

    def _draw_text(self, surf, text, pos, size=16, color=C_TEXT):
        text_surf = self.font_md.render(text, True, color)
        surf.blit(text_surf, pos)

    def _stat_block(self, surf, y_coord, generation, global_best_fitness, best, target_pop, sim_speed, alive_count):
        pad = 18
        curr_y = [y_coord]
        def row(label, value, col=C_TEXT):
            l_surf = self.font_sm.render(label, True, C_TEXT_DIM)
            v_surf = self.font_md.render(str(value), True, col)
            surf.blit(l_surf, (pad, curr_y[0]))
            surf.blit(v_surf, (self.ui_width - v_surf.get_width() - pad, curr_y[0]))
            curr_y[0] += 22
        row("Generation",   generation,                    C_ACCENT)
        row("Target Pop",   target_pop,                    (100, 200, 255))
        row("Global Best",  f"{global_best_fitness:.0f}",  (255, 220, 50))
        if best:
            row("Gen Best",    f"{best.fitness:.0f}")
            row("Sim Speed",   f"{sim_speed}x", (100, 255, 100))
            row("Max Speed",   f"{config.MAX_SPEED:.1f}", (255, 100, 100))
            row("Speed",       f"{best.speed:.2f}")
            row("Checkpoints", best.checkpoints_passed)
        return curr_y[0] + 8

    def _draw_nn(self, surf, y_start: int, car) -> int:
        pad      = 20
        nn       = car.brain
        layers   = NN_LAYERS
        n_layers = len(layers)
        panel_w  = self.ui_width - 2 * pad
        panel_h  = min(320, self.h - y_start - 30)
        label = self.font_sm.render("Neural Network", True, C_TEXT_DIM)
        surf.blit(label, (pad, y_start));  y_start += 20
        col_xs = [pad + int(panel_w * (i / (n_layers - 1)))
                  for i in range(n_layers)]
        node_positions: list[list[tuple]] = []
        for li, (lx, n_nodes) in enumerate(zip(col_xs, layers)):
            spacing = panel_h / (n_nodes + 1)
            positions = [(lx, int(y_start + spacing * (ni + 1)))
                         for ni in range(n_nodes)]
            node_positions.append(positions)
        for li in range(n_layers - 1):
            w_matrix = nn.weights[li]
            for oi, out_pos in enumerate(node_positions[li + 1]):
                for ii, in_pos in enumerate(node_positions[li]):
                    w = w_matrix[oi][ii]
                    col = C_NN_POS if w >= 0 else C_NN_NEG
                    thickness = max(1, min(4, int(abs(w) * 2.5)))
                    alpha     = min(255, int(abs(w) * 180 + 40))
                    blended   = _blend(col, alpha, bg=(12, 12, 22))
                    pygame.draw.line(surf, blended, in_pos, out_pos, thickness)
        r = 10
        for li, positions in enumerate(node_positions):
            for ni, (nx, ny) in enumerate(positions):
                pygame.draw.circle(surf, C_NN_NODE, (nx, ny), r)
                pygame.draw.circle(surf, C_ACCENT,   (nx, ny), r, 2)
                if li == 0:
                    norm_speed = car.speed / config.MAX_SPEED
                    idx = car.next_cp_idx % len(car.track.checkpoints)
                    mx, my = car.track.checkpoint_midpoint(idx)
                    target_angle = math.atan2(my - car.y, mx - car.x)
                    angle_diff = (target_angle - car.angle + math.pi) % (2 * math.pi) - math.pi
                    norm_angle = (angle_diff / math.pi) * 0.5 + 0.5
                    inputs = car.sensor_readings + [norm_speed, norm_angle]
                    if ni < len(inputs):
                        val  = inputs[ni]
                        fill = _blend(C_ACCENT, int(min(1.0, max(0.0, val)) * 220))
                        pygame.draw.circle(surf, fill, (nx, ny), r - 2)
        ly = y_start + panel_h + 8
        pygame.draw.line(surf, C_NN_POS, (pad, ly),      (pad + 20, ly), 2)
        surf.blit(self.font_sm.render("positive", True, C_TEXT_DIM), (pad + 24, ly - 6))
        pygame.draw.line(surf, C_NN_NEG, (pad + 110, ly), (pad + 130, ly), 2)
        surf.blit(self.font_sm.render("negative", True, C_TEXT_DIM), (pad + 134, ly - 6))
        return ly + 18
