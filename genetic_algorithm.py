import random
from config import NUM_CARS, NUM_ELITE, MUTATION_RATE, MUTATION_STRENGTH
from neural_network import NeuralNetwork

class GeneticAlgorithm:
    @staticmethod
    def select_elites(cars) -> list:
        return sorted(cars, key=lambda c: c.fitness, reverse=True)[:NUM_ELITE]

    @staticmethod
    def crossover(p1: NeuralNetwork, p2: NeuralNetwork) -> NeuralNetwork:
        flat1 = p1.get_flat()
        flat2 = p2.get_flat()
        child_flat = [a if random.random() < 0.5 else b
                      for a, b in zip(flat1, flat2)]
        child = NeuralNetwork(p1.layer_sizes)
        child.set_flat(child_flat)
        return child

    @staticmethod
    def mutate(brain: NeuralNetwork) -> NeuralNetwork:
        flat = brain.get_flat()
        mutated = [w + random.gauss(0, MUTATION_STRENGTH)
                   if random.random() < MUTATION_RATE else w
                   for w in flat]
        brain.set_flat(mutated)
        return brain

    def evolve(self, cars, pop_size=NUM_CARS) -> list[NeuralNetwork]:
        elites = self.select_elites(cars)
        new_brains: list[NeuralNetwork] = []
        for e in elites:
            if len(new_brains) < pop_size:
                new_brains.append(e.brain.clone())
        while len(new_brains) < pop_size:
            if len(elites) >= 2:
                p1, p2 = random.sample(elites, 2)
            elif elites:
                p1 = p2 = elites[0]
            else:
                p1 = p2 = cars[0]
            child = self.crossover(p1.brain, p2.brain)
            child = self.mutate(child)
            new_brains.append(child)
        return new_brains[:pop_size]
