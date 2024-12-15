import pygame


class Ground:
    def __init__(self, height, screen_width, screen_height, color=(0, 0, 255)):
        self.height = height
        self.width = screen_width
        self.color = color
        self.dimensions = (0, screen_height - height, screen_width, height)

        self.pos1 = [0, screen_height - height]
        self.pos2 = [screen_width, screen_height - height]

        self.sprite = pygame.image.load("sprites/ground.png")
        self.sprite = pygame.transform.scale(
            self.sprite, (self.sprite.get_size()[0], height)
        )

        # self.collision_rect = pygame.Rect(
        #     self.dimensions[0], self.dimensions[1], self.dimensions[2], self.dimensions[3])
        # self.collision_rect = pygame.Rect(self.sprite.get_rect(
        #     topleft=(self.dimensions[0], self.dimensions[1])))
        self.collision_rect = pygame.Rect(self.dimensions)

    # def update(self, surface):
    #     # pygame.draw.rect(surface, self.color, self.dimensions)
    #     surface.blit(self.sprite, (self.dimensions[0], self.dimensions[1]))

    def update(self, surface):
        surface.blit(self.sprite, self.pos1)
        surface.blit(self.sprite, self.pos2)

        if self.pos1[0] + self.width < 0:
            self.pos1[0] = self.pos1[0] + self.width
        if self.pos2[0] + self.width < 0:
            self.pos2[0] = self.pos1[0] + self.width

        self.pos1[0] -= 7
        self.pos2[0] -= 7


class DoublePipe:
    def __init__(
        self, screen_width, screen_height, gap_pos, gap_height=150, color=(0, 255, 0)
    ):
        self.pos = [screen_width, 0]
        self.color = color
        self.thickness = 75
        self.gap_pos = gap_pos
        self.gap_height = gap_height

        self.up_pipe_rect = [self.pos[0], 0, self.thickness, gap_pos]
        self.down_pipe_rect = [
            self.pos[0],
            gap_pos + gap_height,
            self.thickness,
            screen_height,
        ]
        self.up_collision_rect = pygame.Rect(
            self.up_pipe_rect[0],
            self.up_pipe_rect[1],
            self.up_pipe_rect[2],
            self.up_pipe_rect[3],
        )
        self.down_collision_rect = pygame.Rect(
            self.down_pipe_rect[0],
            self.down_pipe_rect[1],
            self.down_pipe_rect[2],
            self.down_pipe_rect[3],
        )

        self.up_sprite = pygame.transform.rotate(
            pygame.image.load("sprites/pipe.png"), 180
        )
        self.down_sprite = pygame.image.load("sprites/pipe.png")
        # self.up_sprite = pygame.transform.scale(
        #     self.up_sprite, (self.thickness, gap_pos))
        # self.down_sprite = pygame.transform.scale(
        #     self.down_sprite, (self.thickness, screen_height - (gap_pos + gap_height)))
        self.up_sprite = pygame.transform.scale(
            self.up_sprite, (self.thickness, self.up_sprite.get_size()[1])
        )
        self.down_sprite = pygame.transform.scale(
            self.down_sprite, (self.thickness, self.down_sprite.get_size()[1])
        )

        self.has_given_score = False
        self.has_given_score_to = []
        self.is_on_screen = True

    def get_gap_x_and_y(self):
        return [self.pos[0], self.gap_pos]

    def update(self, surface):
        self.up_pipe_rect[0] = self.pos[0]
        self.down_pipe_rect[0] = self.pos[0]
        self.up_collision_rect = pygame.Rect(
            self.up_pipe_rect[0],
            self.up_pipe_rect[1],
            self.up_pipe_rect[2],
            self.up_pipe_rect[3],
        )
        self.down_collision_rect = pygame.Rect(
            self.down_pipe_rect[0],
            self.down_pipe_rect[1],
            self.down_pipe_rect[2],
            self.down_pipe_rect[3],
        )

        # pygame.draw.rect(surface, self.color, self.up_pipe_rect)
        # pygame.draw.rect(surface, self.color, self.down_pipe_rect)
        surface.blit(
            self.up_sprite, (self.pos[0], self.gap_pos - self.up_sprite.get_size()[1])
        )
        surface.blit(self.down_sprite, (self.pos[0], self.gap_pos + self.gap_height))

        if self.pos[0] + self.thickness + 500 < 0:
            self.is_on_screen = False

        self.pos[0] -= 7
