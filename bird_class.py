from math import atan2, pi

import pygame
from pygame.locals import K_SPACE


class Bird:
    def __init__(self, sprite, x=20, y=100, size=10, flap_force=15):
        self.size = size
        self.flap_force = flap_force

        self.pos = [x, y]
        self.vel = [0, 0]
        self.vel_y_lim = -15

        self.is_dead = False
        self.time_survived = 0
        self.score = 0

        # self.color = (200, 200, 0)
        self.sprite = pygame.transform.scale(
            pygame.image.load(sprite), (size * 4, size * 4)
        )
        self.sprite_rect = self.sprite.get_rect(topleft=self.pos)
        self.collision_rect = pygame.Rect(self.sprite.get_rect(topleft=self.pos))

        # Neural net, genetic algorithm stuff
        self.fitness = 0
        self.net = None

    def flap(self):
        self.vel[1] -= self.flap_force
        self.vel[1] = max(self.vel[1], self.vel_y_lim)

    def fall(self, gravity=1):
        self.vel[1] += gravity
        self.pos[1] += self.vel[1]

    def update(self, surface):
        self.fall()
        self.collision_rect = pygame.Rect(self.sprite.get_rect(topleft=self.pos))

        drawing_position = (int(self.pos[0]), int(self.pos[1]))
        # pygame.draw.rect(surface, (255, 0, 0),
        #                  self.collision_rect)  # draw hitbox
        rotated_sprite = pygame.transform.rotate(
            self.sprite, -atan2(self.vel[1], 20) * 180 / pi
        )
        surface.blit(rotated_sprite, drawing_position)

        if self.pos[1] < 0:
            self.is_dead = True
            if self.score == 0:
                self.time_survived // 2

    def get_next_double_pipe(self, double_pipe_list):
        """Gets the next double pipe from a list of position ordered Double_pipe objects"""
        for double_pipe in double_pipe_list:
            if double_pipe.pos[0] + double_pipe.thickness > self.pos[0]:
                return double_pipe

    def sensors(self, double_pipe):
        """returns the y position of the bird and the x and y position of the gap of the given double pipe"""
        gap_x_and_y = double_pipe.get_gap_x_and_y()
        # distance_to_pipe_start = gap_x_and_y[0] - self.pos[0]
        # distance_to_pipe_end = distance_to_pipe_start + double_pipe.thickness
        # return [self.pos[1], distance_to_pipe_start, distance_to_pipe_end, gap_x_and_y[1]]
        return (
            [self.pos[1], self.vel[1]]
            + gap_x_and_y
            + [
                gap_x_and_y[0] + double_pipe.thickness,
                gap_x_and_y[1] + double_pipe.gap_height,
            ]
        )

    def calculate_fitness(self):
        self.fitness = self.time_survived**2 + (self.score * 300) ** 2
        return self.fitness

    def clone(self):
        pass
