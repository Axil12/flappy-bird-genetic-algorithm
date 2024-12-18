import random
import types

import pygame
import numpy as np
from tqdm import tqdm

from bird_class import Bird
from neural_network import NeuralNework


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

        self.birds = []
        for i in range(population_size):
            bird = Bird(self.bird_sprite)
            bird.net = NeuralNework(
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
        child.net = NeuralNework.crossover(parent_1.net, parent_2.net)
        return child

    def mutate(self, mutation_rate: float) -> None:
        for bird in self.birds:
            bird.net.mutate(mutation_rate)

    def natural_selection(
        self, mutation_rate: float = 0.01, allow_clone: bool = False
    ) -> None:
        new_bird_list = []
        self.calculate_fitness_sum()

        for i in tqdm(
            range(len(self.birds)),
            desc=f"Reproduction | Generation {self.gen:04d}",
            ncols=80,
        ):
            parents = self.select_parents(allow_clone=allow_clone)
            child = self.reproduce(parents[0], parents[1])
            new_bird_list.append(child)

        self.birds = new_bird_list
        self.mutate(mutation_rate=0.01)
        self.gen += 1
