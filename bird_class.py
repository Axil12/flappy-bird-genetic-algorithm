import random
from math import atan2, pi

import numpy as np
import pygame
from pygame.locals import K_SPACE

from obstacles import DoublePipe


class Bird:
    def __init__(
        self,
        sprite: str,
        x: int = 100,
        y: int = 100,
        size: int = 10,
        flap_force: float = 15,
    ):
        self.size = size
        self.flap_force = flap_force

        self.pos = [x, y]
        self.vel = [0, 0]
        self.vel_y_lim = -15

        self.is_dead = False
        self.time_survived = 0
        self.time_since_death = 0
        self.score = 0

        # self.color = (200, 200, 0)
        self.sprite = pygame.transform.scale(
            pygame.image.load(sprite), (size * 4, size * 4)
        )
        self.sprite_alt = pygame.transform.scale(
            pygame.image.load("sprites/zubat_shiny.png"), (size * 4, size * 4)
        )
        self.sprite_rect = self.sprite.get_rect(topleft=self.pos)
        self.collision_rect = pygame.Rect(self.sprite.get_rect(topleft=self.pos))

        # Neural net, genetic algorithm stuff
        self.fitness = 0
        self.net = None

    def flap(self):
        self.vel[1] -= self.flap_force
        self.vel[1] = max(self.vel[1], self.vel_y_lim)

    def fall(self, gravity: float = 1):
        self.vel[1] += gravity
        self.pos[1] += self.vel[1]

    def death_animation(self, surface: pygame.Surface) -> None:
        if self.time_since_death >= 50:
            return
        if self.time_since_death == 0:
            self.vel[1] = -20

        self.pos[0] -= 9

        self.fall(gravity=2)
        rotated_sprite = pygame.transform.rotate(
            self.sprite, 20 * self.time_since_death
        )
        surface.blit(rotated_sprite, self.pos)

        self.time_since_death += 1

    def update(self, surface: pygame.Surface) -> None:
        self.fall()
        self.collision_rect = pygame.Rect(self.sprite.get_rect(topleft=self.pos))

        drawing_position = (int(self.pos[0]), int(self.pos[1]))
        # pygame.draw.rect(surface, (255, 0, 0),
        #                  self.collision_rect)  # draw hitbox
        rotated_sprite = pygame.transform.rotate(
            self.sprite, -atan2(self.vel[1], 20) * 180 / pi
        )
        surface.blit(rotated_sprite, drawing_position)

        if self.pos[1] + self.sprite.get_height() < 0:
            self.is_dead = True
            if self.score == 0:
                self.time_survived // 2

    def get_next_double_pipe(self, double_pipe_list: list[DoublePipe]):
        """Gets the next double pipe from a list of position ordered Double_pipe objects"""
        for double_pipe in double_pipe_list:
            if double_pipe.pos[0] + double_pipe.thickness > self.pos[0]:
                return double_pipe

    def sensors(self, double_pipe: DoublePipe, screen: pygame.Surface) -> list[float]:
        """returns the y position of the bird and the x and y position of the gap of the given double pipe"""
        # We need these informations in order to normalize the data
        screen_width, screen_height = screen.get_size()

        gap_x_and_y = double_pipe.get_gap_x_and_y()
        y_pos = self.pos[1] / screen_height
        y_vel = self.vel[1] / abs(self.vel_y_lim)
        gap_x_front = gap_x_and_y[0] / screen_width
        gap_x_back = gap_x_front + double_pipe.thickness / screen_width
        gap_top = gap_x_and_y[1] / screen_height
        gap_bottom = gap_top + double_pipe.gap_height / screen_height

        # To test if the neural network can learn that this input is useless
        # gap_bottom = random.uniform(-150, 150)

        return [y_pos, y_vel, gap_x_front, gap_x_back, gap_top, gap_bottom]

    def calculate_fitness(self):
        # self.fitness = self.time_survived**2 + (self.score * 300) ** 2
        self.fitness = self.time_survived**2
        return self.fitness

    def clone(self):
        pass

    def linear_output(self, x: np.array) -> float:
        """
        It turns out, Flappy Bird can be solved with a linear equation.
        It this current implementation of the game, the following factors
        can solve the game and entierely replace a neural network.
        """
        # factors = np.array(
        #     [
        #         0.41484223,
        #         0.09051711,
        #         -0.42681699,
        #         0.38675124,
        #         -0.40437329,
        #         -0.03219503,
        #         0.40897791,
        #     ]
        # )
        factors = np.array([[0.4, 0.1, -0.4, 0.4, -0.4, -0.03, 0.4]])
        x = np.concatenate([x, [[1]]], axis=0)
        return np.dot(factors, x)
