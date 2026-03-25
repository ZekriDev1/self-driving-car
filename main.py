import sys
import pygame
import config
from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, FPS, NUM_CARS, SIM_SPEED_MULT,
    ZOOM_STEP, MIN_ZOOM, MAX_ZOOM
)
from track            import Track
from car              import Car
from genetic_algorithm import GeneticAlgorithm
from renderer         import Renderer

import json
import os

BEST_BRAIN_FILE = "best_brain.json"

def save_brain(brain):
    with open(BEST_BRAIN_FILE, "w") as f:
        json.dump(brain.get_flat(), f)

def load_brain_flat():
    if os.path.exists(BEST_BRAIN_FILE):
        try:
            with open(BEST_BRAIN_FILE, "r") as f:
                content = f.read().strip()
                if not content: return None
                return json.loads(content)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {BEST_BRAIN_FILE} ({e}). Starting fresh.")
            return None
    return None

def make_generation(track: Track, brains=None, pop_size=NUM_CARS) -> list[Car]:
    if brains is None:
        saved_flat = load_brain_flat()
        if saved_flat:
            print("Loading saved best brain...")
            dummy = Car(track)
            first_brain = dummy.brain 
            if len(saved_flat) == len(first_brain.get_flat()):
                first_brain.set_flat(saved_flat)
            else:
                print(f"Saved brain architecture mismatch ({len(saved_flat)} weights vs expected {len(first_brain.get_flat())}). Starting fresh.")
                return [Car(track) for _ in range(pop_size)]
            
            ga = GeneticAlgorithm()
            brains = [ga.mutate(first_brain.clone()) for _ in range(pop_size)]
            brains[0] = first_brain
        else:
            return [Car(track) for _ in range(pop_size)]
    return [Car(track, brain=b) for b in brains]

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("Self-Driving Evolution (R: New Track, S: Save, 1: Single, +/-: Pop, [/]: Sim Spd, UP/DOWN: Max Spd)")
    clock  = pygame.time.Clock()

    track    = Track()
    ga       = GeneticAlgorithm()
    renderer = Renderer(screen)

    generation   = 1
    global_best_fitness = 0.0
    target_pop   = NUM_CARS
    cars         = make_generation(track, pop_size=target_pop)
    completed_cars = []
    reset_timer  = 0
    sim_speed_scale = SIM_SPEED_MULT
    running      = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            if event.type == pygame.VIDEORESIZE:
                renderer.handle_resize(event.w, event.h)

            if event.type == pygame.MOUSEWHEEL:
                zoom_dir = event.y
                renderer.camera.zoom = max(MIN_ZOOM, min(MAX_ZOOM, renderer.camera.zoom + zoom_dir * ZOOM_STEP))

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    for c in cars: c.alive = False
                if event.key == pygame.K_r:
                    track = Track()
                    generation = 1
                    cars = make_generation(track, pop_size=target_pop)
                if event.key == pygame.K_s:
                    sorted_cars = sorted(cars, key=lambda c: c.fitness, reverse=True)
                    if sorted_cars:
                        save_brain(sorted_cars[0].brain)
                        print("Best brain saved!")
                
                if event.key == pygame.K_1:
                    target_pop = 1
                    for c in cars: c.alive = False
                if event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    target_pop = min(200, target_pop + 10)
                if event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    target_pop = max(1, target_pop - 10)
                if event.key == pygame.K_LEFTBRACKET:
                    sim_speed_scale = max(1, sim_speed_scale - 1)
                if event.key == pygame.K_RIGHTBRACKET:
                    sim_speed_scale = min(20, sim_speed_scale + 1)
                
                if event.key == pygame.K_UP:
                    config.MAX_SPEED = min(15.0, config.MAX_SPEED + 0.5)
                    config.ACCELERATION = config.MAX_SPEED * (0.14 / 4.5)
                if event.key == pygame.K_DOWN:
                    config.MAX_SPEED = max(1.0, config.MAX_SPEED - 0.5)
                    config.ACCELERATION = config.MAX_SPEED * (0.14 / 4.5)

                if event.key == pygame.K_n:
                    new_car = Car(track)
                    new_car.x, new_car.y = track.start_pos
                    new_car.angle = track.start_heading
                    cars.append(new_car)

        for _ in range(sim_speed_scale):
            still_alive = []
            for car in cars:
                car.update()
                if car.alive:
                    still_alive.append(car)
                else:
                    completed_cars.append(car)
            cars = still_alive

        all_cars_pool = cars + completed_cars
        
        alive_any = len(cars) > 0
        if all_cars_pool:
            gen_best = max(all_cars_pool, key=lambda c: c.fitness)
            if gen_best.fitness > global_best_fitness:
                global_best_fitness = gen_best.fitness

        renderer.render(track, cars, generation, global_best_fitness, target_pop, sim_speed_scale, all_cars_pool)
        pygame.display.flip()
        clock.tick(FPS)

        if not alive_any:
            reset_timer += 1
            if reset_timer > 60:
                all_cars_this_gen = cars + completed_cars
                new_brains = ga.evolve(all_cars_this_gen, pop_size=target_pop)
                generation += 1
                cars = make_generation(track, brains=new_brains, pop_size=target_pop)
                completed_cars = []
                reset_timer = 0
        else:
            reset_timer = 0

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
