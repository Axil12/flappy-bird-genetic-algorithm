from random import randint

import pygame
import numpy as np
from tqdm import tqdm

from neural_network import NeuralNework
from bird_population import BirdPopulation
from bird_class import Bird
from obstacles import Ground, DoublePipe
from utils import draw_neural_network


def main():
    screen_width = 800
    screen_height = 600
    background = pygame.transform.scale(
        pygame.image.load("sprites/background.png"), (screen_width, screen_height)
    )

    net_dims = (6, (5, 3), 2)
    pop_size = 300
    nb_generations = 1000

    pop = BirdPopulation(
        pop_size, net_dims, NeuralNework.leaky_relu, "sprites/zubat.png"
    )
    for _ in range(nb_generations):
        screen = pygame.display.set_mode((screen_width, screen_height))
        clock = pygame.time.Clock()

        ground = Ground(30, screen_width, screen_height)
        double_pipes = [
            DoublePipe(
                screen_width,
                screen_height,
                randint(int(screen_height * 0.1), int(screen_height * 0.7)),
            )
        ]

        tick_passed = 0
        is_running = True
        while is_running:
            clock.tick(30)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    isRunning = False
            screen.blit(background, (0, 0))

            for double_pipe in double_pipes:
                if not double_pipe.is_on_screen:
                    double_pipes.remove(double_pipe)
                double_pipe.update(screen)

            has_draw_lines = False

            ground.update(screen)
            for bird in pop.birds:
                if bird.is_dead:
                    continue

                if not has_draw_lines:
                    has_draw_lines = True
                    s = bird.sensors(bird.get_next_double_pipe(double_pipes))
                    pygame.draw.line(screen, (255, 0, 0), bird.pos, (s[2], s[4]))
                    pygame.draw.line(screen, (0, 255, 0), bird.pos, (s[3], s[4]))
                    pygame.draw.line(screen, (0, 0, 255), bird.pos, (s[2], s[5]))
                    outputs = bird.net.get_layers_outputs(
                        np.array(s).T.reshape((len(s), 1))
                    )
                    draw_neural_network(screen, (520, 30), 300, 200, bird.net, outputs)

                bird.time_survived += 1
                net_input = bird.sensors(bird.get_next_double_pipe(double_pipes))
                net_input = np.array(net_input).T
                net_input = net_input.reshape((net_input.shape[0], 1))
                net_output = bird.net.output(net_input)
                if np.argmax(net_output) == 1:
                    bird.flap()
                bird.update(screen)

                if bird.collision_rect.colliderect(ground.collision_rect):
                    bird.is_dead = True
                    if bird.score == 0:
                        bird.time_survived // 2

                for double_pipe in double_pipes:
                    if bird.collision_rect.colliderect(
                        double_pipe.up_collision_rect
                    ) or bird.collision_rect.colliderect(
                        double_pipe.down_collision_rect
                    ):
                        bird.is_dead = True

                    if bird not in double_pipe.has_given_score_to:
                        if bird.pos[0] > double_pipe.pos[0] + double_pipe.thickness:
                            double_pipe.has_given_score_to.append(bird)
                            bird.score += 1

            if tick_passed >= 50:
                new_douple_pipe = DoublePipe(
                    screen_width,
                    screen_height,
                    randint(int(screen_height * 0.1), int(screen_height * 0.7)),
                )
                double_pipes.append(new_douple_pipe)
                tick_passed = 0

            current_best_score = max([bird.score for bird in pop.birds])
            birds_alive = sum([not bird.is_dead for bird in pop.birds])
            pygame.display.set_caption(
                "Flappy Bird | Birds alive : %03d | current best score : %d"
                % (birds_alive, current_best_score)
            )
            pygame.display.update()

            tick_passed += 1
            if birds_alive == 0:
                break

        print()
        print("Best score of generation %04d : %d" % (pop.gen, current_best_score))
        pop.natural_selection()

    pygame.quit()


if __name__ == "__main__":
    main()
