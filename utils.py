from math import exp

import pygame
import numpy as np

from bird_class import Bird
from obstacles import DoublePipe

def draw_neural_network(surface, position, width, height, net, layer_outputs):
    n_cols = len(net.dims)
    n_rows = max(net.dims)

    r = 12  # radius of the circles drawn

    # Drawing the lines
    for col in range(n_cols - 1):
        w_max_abs = max(abs(net.weight_min_val), abs(net.weight_max_val))
        for row in range(len(layer_outputs[col])):
            pos_1 = (
                position[0] + col * width // n_cols + r,
                position[1] + row * height // n_rows + r,
            )
            for next_row in range(net.dims[col + 1]):
                pos_2 = (
                    position[0] + (col + 1) * width // n_cols + r,
                    position[1] + next_row * height // n_rows + r,
                )

                w = net.weights[col][next_row][row]
                value = int(abs(w) / w_max_abs * 255)
                line_width = int(value / 255 * 5 + 1)
                if abs(w) / w_max_abs < 0.02:
                    line_width = 0
                line_color = (
                    (0, value, 0) if w > 0 else (value, 0, 0)
                )  # Green if weight > 0, else Red
                pygame.draw.line(surface, line_color, pos_1, pos_2, width=line_width)

    font = pygame.font.Font("freesansbold.ttf", 10)
    # Drawing circles
    for col in range(n_cols):
        layer_val_min = np.min(layer_outputs[col])
        layer_val_max = np.max(layer_outputs[col])
        for row in range(len(layer_outputs[col])):
            center = (
                position[0] + col * width // n_cols + r,
                position[1] + row * height // n_rows + r,
            )

            pygame.draw.circle(surface, (30, 30, 30), center, r + 2)
            neuron_value = layer_outputs[col][row]
            if neuron_value < 0:
                value = neuron_value / min(-0.001, layer_val_min)
                circle_color = (int(value * 255), 0, 0)
            else:
                value = neuron_value / max(0.001, layer_val_max)
                circle_color = (0, int(value * 255), 0)
            pygame.draw.circle(surface, circle_color, center, r)
            text = font.render(
                f"{float(neuron_value):.2f}", True, (0, 0, 255)  # antialiasing
            )
            text_rect = text.get_rect()
            text_rect.center = center
            surface.blit(text, text_rect)


def draw_info(
    screen: pygame.Surface, bird: Bird, double_pipes: list[DoublePipe]
) -> None:
    bird.sprite = bird.sprite_alt
    next_double_pipe = bird.get_next_double_pipe(double_pipes)
    s = bird.sensors(next_double_pipe, screen)
    bird_center = (
        bird.pos[0] + bird.sprite.get_width() // 2,
        bird.pos[1] + bird.sprite.get_height() // 2,
    )
    w, h = screen.get_size()
    pygame.draw.line(screen, (255, 0, 0), bird_center, (s[2] * w, s[4] * h))
    pygame.draw.line(screen, (0, 255, 0), bird_center, (s[3] * w, s[4] * h))
    pygame.draw.line(screen, (0, 0, 255), bird_center, (s[2] * w, s[5] * h))
    pygame.draw.line(screen, (255, 255, 0), bird_center, (s[3] * w, s[5] * h))
    outputs = bird.net.get_layers_outputs(np.array(s).T.reshape((len(s), 1)))
    draw_neural_network(
        screen,
        (w - int(0.4 * w) + 30, 30),
        int(0.4 * w),
        int(0.3 * h),
        bird.net,
        outputs,
    )
