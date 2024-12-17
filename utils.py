import pygame
import numpy as np

from neural_network import NeuralNework


def draw_neural_network(surface, position, width, height, net, layer_outputs):
    layer_dims = [net.nb_inputs] + list(net.hidden_layers) + [net.nb_outputs]
    weights = [net.whi] + list(net.whh_list) + [net.woh]

    n_cols = len(layer_dims)
    n_rows = max(layer_dims)

    r = 10  # radius of the circles drawn

    # Drawing the lines
    for col in range(n_cols - 1):
        w_min = np.min(weights[col])
        w_max = np.max(weights[col])
        w_max_abs = max(abs(w_min), abs(w_max))
        for row in range(len(layer_outputs[col])):
            pos_1 = (
                position[0] + col * width // n_cols + r,
                position[1] + row * height // n_rows + r,
            )
            for next_row in range(layer_dims[col + 1]):
                pos_2 = (
                    position[0] + (col + 1) * width // n_cols + r,
                    position[1] + next_row * height // n_rows + r,
                )

                w = weights[col][next_row][row]
                value = int(abs(w) / w_max_abs * 255)
                line_width = int(value / 255 * 5 + 1)
                if value == 0:
                    line_width = 0
                line_color = (0, value, 0) if w > 0 else (value, 0, 0)
                pygame.draw.line(surface, line_color, pos_1, pos_2, width=line_width)

    # Drawing circles
    for col in range(n_cols):
        o_min = np.min(layer_outputs[col])
        o_max = np.max(layer_outputs[col])
        for row in range(len(layer_outputs[col])):
            center = (
                position[0] + col * width // n_cols + r,
                position[1] + row * height // n_rows + r,
            )

            if o_max == o_min:
                alpha = 0
            else:
                alpha = layer_outputs[col][row]
                alpha = int((alpha - o_min) / (o_max - o_min) * 255)

            pygame.draw.circle(surface, (0, 255, 0), center, r + 1)
            pygame.draw.circle(surface, (0, 0, alpha), center, r)
