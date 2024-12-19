import random
import types

import pygame
import numpy as np
from tqdm import tqdm

from bird_class import Bird
from neural_network import NeuralNetwork


class BirdPopulation:
    def __init__(
        self,
        population_size: int,
        neural_network_dims: tuple[int, tuple[int], int],
        neural_network_activation: types.FunctionType,
        bird_sprite: str,
    ):

        self.population_size = population_size
        self.bird_sprite = bird_sprite
        self.bird_neural_network_dims = neural_network_dims
        self.bird_neural_network_activation = neural_network_activation

        self.birds = []
        for i in range(population_size):
            bird = Bird(self.bird_sprite)
            bird.net = NeuralNetwork(
                neural_network_dims, activation=neural_network_activation
            )
            self.birds.append(bird)

        self.gen = 1
        self.fitness_sum = 0

    def calculate_fitnesses(self) -> None:
        for bird in self.birds:
            bird.calculate_fitness()

    def calculate_fitness_sum(self) -> float:
        self.fitness_sum = sum([bird.calculate_fitness() for bird in self.birds])
        return self.fitness_sum

    def select_parents(self, allow_clone: bool = False) -> tuple[Bird, Bird]:
        rand = random.uniform(0, self.fitness_sum)
        running_sum = 0
        for bird in self.birds:
            running_sum += bird.fitness
            if running_sum > rand:
                parent_1 = bird
                break

        if allow_clone:
            rand = random.uniform(0, self.fitness_sum)
            running_sum = 0
            for bird in self.birds:
                running_sum += bird.fitness
                if running_sum > rand:
                    parent_2 = bird
                    break
        else:
            parent_2 = parent_1
            while parent_2 == parent_1:
                rand = random.uniform(0, self.fitness_sum)
                running_sum = 0
                for bird in self.birds:
                    running_sum += bird.fitness
                    if running_sum > rand:
                        parent_2 = bird
                        break

        return parent_1, parent_2

    def reproduce(self, parent_1, parent_2):
        child = Bird(self.bird_sprite)
        child.net = NeuralNetwork.crossover(parent_1.net, parent_2.net)
        return child

    def natural_selection(
        self,
        mutation_rate: float = 0.01,
        allow_clone: bool = False,
        redraw_rate: float = 0.05,
        keep_champion: bool = True,
    ) -> None:
        self.calculate_fitness_sum()
        new_bird_list = []

        best_bird = max(self.birds, key=lambda b: b.fitness)

        for i in tqdm(
            range(len(self.birds)),
            desc=f"Reproduction | Generation {self.gen:04d}",
            ncols=80,
        ):
            parents = self.select_parents(allow_clone=allow_clone)
            child = self.reproduce(parents[0], parents[1])
            child.fitness = (parents[0].fitness + parents[1].fitness) / 2
            new_bird_list.append(child)

        # We delete the n weakest children and replace them by a completely new bird
        if redraw_rate > 0:
            n_redraws = int(redraw_rate * len(new_bird_list))
            new_bird_list = sorted(new_bird_list, key=lambda b: b.fitness)
            min_fitness = new_bird_list[0].fitness
            for i in range(n_redraws):
                bird = Bird(self.bird_sprite)
                bird.net = NeuralNetwork(
                    self.bird_neural_network_dims,
                    activation=self.bird_neural_network_activation,
                )
                new_bird_list.append(bird)
            new_bird_list = new_bird_list[n_redraws:]

        for bird in new_bird_list:
            bird.net.mutate(mutation_rate=mutation_rate)

        # To keep the best of the previous generation to see if he is still the best
        if keep_champion:
            champion_bird = Bird(self.bird_sprite)
            champion_bird.sprite = best_bird.sprite_alt
            champion_bird.net = best_bird.net

            wost_bird_of_new_gen = min(new_bird_list, key=lambda b: b.fitness)
            idx = new_bird_list.index(wost_bird_of_new_gen)
            new_bird_list[idx] = champion_bird

        assert len(new_bird_list) == len(self.birds)

        self.birds = new_bird_list
        self.gen += 1
