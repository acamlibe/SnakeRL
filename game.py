import random
from enum import Enum
from collections import namedtuple
import numpy as np
import pygame

pygame.init()

font = pygame.font.Font('arial.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
HEAD = (40, 176, 140)
BLACK = (0,0,0)

BLOCK_SIZE = 20
SPEED = 600

CLOCK_WISE = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]

class SnakeGameAI:

    def __init__(self, w=1280, h=720):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.direction = Direction.RIGHT
        self.food = None
        self.head = Point(self.w/2, self.h/2)
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self.frame_iteration = 0
        self._place_food()


    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE 
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()


    def play_step(self, action):
        self.frame_iteration += 1

        self._move(action)
        self.snake.insert(0, self.head)

        reward = 0
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            reward = -15
            return reward, True, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            reward = -0.05
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, False, self.score


    def is_collision(self, point=None):
        if point is None:
            point = self.head

        # hits boundary
        if (point.x > self.w - BLOCK_SIZE or 
            point.x < 0 or 
            point.y > self.h - BLOCK_SIZE or 
            point.y < 0):
            return True

        # hits itself
        if point in self.snake[1:]:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)
        is_head = False
        for pt in self.snake:
            rect = pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE)
            if is_head:
                pygame.draw.rect(self.display, WHITE, rect)
            else:
                pygame.draw.rect(self.display, HEAD, rect)
            is_head = True
       
        pygame.draw.rect(self.display,
                         RED,
                         pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text_str = f'Score: {self.score} | Length: {len(self.snake)}'
        text = font.render(text_str, True, WHITE)

        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]
        dir_idx = CLOCK_WISE.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = CLOCK_WISE[dir_idx]
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (dir_idx + 1) % 4
            new_dir = CLOCK_WISE[next_idx]
        else:
            next_idx = (dir_idx - 1) % 4
            new_dir = CLOCK_WISE[next_idx]

        self.direction = new_dir

        x = self.head.x
        y = self.head.y

        match self.direction:
            case Direction.RIGHT:
                x += BLOCK_SIZE
            case Direction.LEFT:
                x -= BLOCK_SIZE
            case Direction.DOWN:
                y += BLOCK_SIZE
            case Direction.UP:
                y -= BLOCK_SIZE

        self.head = Point(x, y)
