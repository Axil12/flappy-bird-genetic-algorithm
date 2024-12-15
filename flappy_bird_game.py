from random import randint

import pygame
import numpy

from bird_class import Bird
from obstacles import Ground, DoublePipe


def play_flappy_bird():
    screen_width = 800
    screen_height = 600
    background = pygame.transform.scale(
        pygame.image.load("sprites/background.png"), (screen_width, screen_height)
    )
    screen = pygame.display.set_mode((screen_width, screen_height))

    bird = Bird(
        sprite="sprites/zubat.png",
        x=int(screen_width * 0.15),
        y=int(screen_height * 0.1),
    )
    ground = Ground(30, screen_width, screen_height)
    double_pipes = [
        DoublePipe(
            screen_width,
            screen_height,
            randint(int(screen_height * 0.3), int(screen_height * 0.7)),
        )
    ]

    clock = pygame.time.Clock()

    tick_passed = 0
    is_running = True
    while is_running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.flap()

        # screen.fill((200, 200, 200))
        screen.blit(background, (0, 0))

        bird.update(screen)
        for double_pipe in double_pipes:
            if not double_pipe.is_on_screen:
                double_pipes.remove(double_pipe)
            double_pipe.update(screen)
        ground.update(screen)

        if bird.collision_rect.colliderect(ground.collision_rect):
            bird.is_dead = True
        for double_pipe in double_pipes:
            if bird.collision_rect.colliderect(
                double_pipe.up_collision_rect
            ) or bird.collision_rect.colliderect(double_pipe.down_collision_rect):
                bird.is_dead = True

        for double_pipe in double_pipes:
            if not double_pipe.has_given_score:
                if bird.pos[0] > double_pipe.pos[0] + double_pipe.thickness:
                    double_pipe.has_given_score = True
                    bird.score += 1

        if tick_passed >= 50:
            new_pipe = DoublePipe(
                screen_width,
                screen_height,
                randint(int(screen_height * 0.3), int(screen_height * 0.7)),
            )
            double_pipes.append(new_pipe)
            tick_passed = 0

        if bird.is_dead:
            print("\nGAME OVER\n")
            is_running = False

        pygame.display.set_caption("Flappy Bird | score : %d" % bird.score)
        pygame.display.update()

        tick_passed += 1

    pygame.quit()


def main():
    play = True
    while play:
        play_flappy_bird()
        playAnotherGame = input("Do you want to play another game ? (y/n) : ")
        if playAnotherGame[0].lower().strip() != "y":
            play = False


if __name__ == "__main__":
    main()
