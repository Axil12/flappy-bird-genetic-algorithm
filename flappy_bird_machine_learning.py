from random import randint

import pygame
import numpy as np
from tqdm import tqdm

from neural_network import NeuralNetwork
from bird_population import BirdPopulation
from bird_class import Bird
from obstacles import Ground, DoublePipe
from utils import draw_neural_network, draw_info


def main():
    pygame.init()

    simulation_speed = 1.0
    screen_width = 800
    screen_height = 600
    background = pygame.transform.scale(
        pygame.image.load("sprites/background.png"), (screen_width, screen_height)
    )

    net_dims = (6, 5, 3, 1)
    # net_dims = (6, 4, 1)
    # net_dims = (6, 1)
    pop_size = 200
    nb_generations = 1000

    pop = BirdPopulation(
        population_size=pop_size,
        neural_network_dims=net_dims,
        # neural_network_activation=NeuralNetwork.step,
        # neural_network_activation=NeuralNetwork.relu,
        # neural_network_activation=NeuralNetwork.leaky_relu,
        # neural_network_activation=NeuralNetwork.sigmoid,
        # neural_network_activation=NeuralNetwork.softplus,
        neural_network_activation=NeuralNetwork.silu,
        # neural_network_activation=NeuralNetwork.gelu,
        # neural_network_activation=NeuralNetwork.elu,
        # neural_network_activation=NeuralNetwork.square,
        # neural_network_activation=NeuralNetwork.linear,
        bird_sprite="sprites/zubat.png",
    )
    for _ in range(nb_generations):
        screen = pygame.display.set_mode((screen_width, screen_height))
        clock = pygame.time.Clock()

        ground = Ground(30, screen_width, screen_height)
        double_pipes = [
            DoublePipe(
                screen_width,
                screen_height,
                randint(int(screen_height * 0.1), int(screen_height * 0.6)),
            )
        ]

        ticks_until_next_pipe = 50
        tick_passed = 0
        is_running = True
        while is_running:
            clock.tick(int(30 * simulation_speed))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    isRunning = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP:
                        simulation_speed = min(4.0, simulation_speed + 0.1)
                    elif event.key == pygame.K_DOWN:
                        simulation_speed = max(0.1, simulation_speed - 0.1)

            screen.blit(background, (0, 0))

            for double_pipe in double_pipes:
                if not double_pipe.is_on_screen:
                    double_pipes.remove(double_pipe)
                double_pipe.update(screen)

            draw_info(screen, [b for b in pop.birds if not b.is_dead][-1], double_pipes)

            ground.update(screen)
            for bird in pop.birds:
                if bird.is_dead:
                    bird.death_animation(screen)
                    continue

                bird.time_survived += 1
                net_input = bird.sensors(
                    bird.get_next_double_pipe(double_pipes), screen
                )
                net_input = np.array(net_input).T
                net_input = net_input.reshape((net_input.shape[0], 1))
                net_output = bird.net(net_input)
                if net_output > 0.5:
                    # if bird.linear_output(net_input) > 0.5:
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

            if tick_passed >= ticks_until_next_pipe:
                ticks_until_next_pipe = 50 + randint(-10, 10)
                new_douple_pipe = DoublePipe(
                    screen_width,
                    screen_height,
                    randint(int(screen_height * 0.1), int(screen_height * 0.6)),
                    gap_height=randint(90, 200),
                )
                double_pipes.append(new_douple_pipe)
                tick_passed = 0

            current_best_score = max([bird.score for bird in pop.birds])
            birds_alive = sum([not bird.is_dead for bird in pop.birds])
            for bird in pop.birds:
                if bird.score == current_best_score:
                    best_bird = bird
            pygame.display.set_caption(
                f"Flappy Bird | {simulation_speed:.1f}x Speed | Birds alive : {birds_alive:03d} | Score : {current_best_score}"
            )
            pygame.display.update()

            tick_passed += 1

            if birds_alive == 0:
                break
            if current_best_score >= 500:
                best_bird.net.save(directory="saved_neural_networks")
                break

        average_score = sum([bird.score for bird in pop.birds]) / len(pop.birds)
        print()
        print(
            f"Best score of generation {pop.gen:04d} : {current_best_score} | Average score : {average_score:.3f}"
        )
        pop.natural_selection(
            mutation_rate=0.02,
            allow_clone=False,
            redraw_rate=0.05,
        )

    pygame.quit()


if __name__ == "__main__":
    main()
