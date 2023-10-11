import pygame
from pygame import Rect, Surface, Vector2
from pygame.event import Event

import numpy as np
from collections import deque
from enum import Enum
from typing import NamedTuple, Sequence, Literal
from dataclasses import dataclass
import weakref

import random
from copy import deepcopy

pygame.init()


class Color(Enum):

    BLACK = (0, 0, 0)
    WHITE = (0xff, 0xff, 0xff)
    RED = (0xff, 0, 0)
    GREEN = (0, 0xff, 0)
    BLUE = (0, 0, 0xff)
    SNAKE_COLOR = (0x2f, 0x7f, 0x2f)
    HEAD_COLOR = (0xf, 0xcf, 0)
    FRUIT_COLOR = (0xff, 0xff, 0)


class Direction(Enum):

    UP = 1
    RIGHT = -2
    DOWN = -1
    LEFT = 2

    @property
    def mask(self):
        mapping = {
            1: Vector2(0, -1),
            -1: Vector2(0, 1),
            2: Vector2(-1, 0),
            -2: Vector2(1, 0),
        }
        return mapping[self.value]


class RenderMode(Enum):
    RGB_IMAGE = 0
    GRAYSCALE_IMAGE = 1
    GRAYSCALE_IMAGE_SHRINKED = 2
    BINARY_VECTOR = 3
    DISTANCED_VECTOR = 4


class Setup:

    BLOCK_SIZE = 32
    GAME_SPEED = 60
    SNAKE_LEN = 3

    def __init__(self, field_size: int):
        self.FIELD_SIZE = field_size
        self.DISPLAY_SIZE = field_size * self.BLOCK_SIZE


@dataclass
class Block:
    coords: Vector2
    size: int = Setup.BLOCK_SIZE
    color: Color = Color.BLACK


class NumpyDriver:

    OHE_4X4 = np.eye(4)

    def __init__(self, game: weakref.ref["Game"]):
        self.__game = game
        self.directions = [d for d in Direction]

    def surface_to_numpy(self, surface: Surface):
        return pygame.surfarray.array3d(surface).transpose((1, 0, 2))

    def action_to_direction(self, current_direction: Direction, action: int):
        index = self.directions.index(current_direction) + action
        if index > 3:
            index = 0
        return self.directions[index]

    def generate_binary_array(self, surface: Surface, direction: int, head_coords: tuple[int, int], apple_coords: tuple[int, int], snake_dead: bool):
        """
        Returns binary vector of size 11, reprsenting boolean values of game state
        [
            danger: [left, straight, right]
            direction_ohe: [left, up, righ, down]
            food_placment: [left, up, righ, down]
        ]
        """

        head_x, head_y = map(int, head_coords)
        apple_x, apple_y = map(int, apple_coords)
        field_size = self.__game().setup.FIELD_SIZE

        if snake_dead:
            # hardcoded solution to show that game is over
            return np.array([*[1]*3,
                             *[0]*4,
                             *[0]*4,
                             ])

        direction_mapping = {
            1: 1,  # up
            -1: 3,  # down
            2: 0,  # left
            -2: 2,  # right
        }

        direction, reverse_direction = direction_mapping[direction], direction_mapping[-direction]
        direction_ohe = self.OHE_4X4[direction]

        walls_near = np.array(
            [head_y, head_x, field_size - 1 - head_x,
                field_size - 1 - head_y]  # [up, left, right, down]
        ) == 0

        frame = self.surface_to_numpy(surface)[::self.__game(
        ).setup.BLOCK_SIZE, ::self.__game().setup.BLOCK_SIZE, 1]

        frame = np.pad(frame, 1, mode="constant",
                       constant_values=Color.SNAKE_COLOR.value[1])
        frame_slice = frame[head_y: head_y + 3,
                            head_x: head_x + 3].flatten()[1::2]

        body_part_near = frame_slice == Color.SNAKE_COLOR.value[1]
        # [up, left, right, down]

        danger_near = walls_near | body_part_near

        danger_near[[0, 1]] = danger_near[[1, 0]]  # [left, up, right, down]

        danger = np.delete(danger_near, reverse_direction)

        distance_to_apple = (apple_x - head_x, apple_y - head_y)

        food_placement = np.zeros((4,))

        if distance_to_apple[0] > 0:
            food_placement[2] = 1
        elif distance_to_apple[0] < 0:
            food_placement[0] = 1

        if distance_to_apple[1] > 0:
            food_placement[3] = 1
        elif distance_to_apple[1] < 0:
            food_placement[1] = 1

        return np.concatenate((danger, direction_ohe, food_placement)).astype("int8")

    def generate_distanced_array(self, surface: Surface, direction: int, head_coords: tuple[int, int], apple_coords: tuple[int, int], snake_dead: bool):
        """
        Returns vector, reprsenting values of game state
        [
            direction_ohe: [left, up, righ, down]
            distances_to_walls: [left, up, righ, down]
            distances_to_apple: [x, y]
            body_part_near: [up_left, up, up_right, left, righ, down_left, down, down_right]
        ]
        """

        head_x, head_y = map(int, head_coords)
        apple_x, apple_y = map(int, apple_coords)
        field_size = self.__game().setup.FIELD_SIZE

        if snake_dead:
            # hardcoded solution to show that game is over
            return np.array([*[0]*4,
                             *[1]*4,
                             *[field_size]*2,
                             *[1]*8
                             ])

        direction_mapping = {
            1: 1,
            -1: 3,
            2: 0,
            -2: 2,
        }

        direction = direction_mapping[direction]
        direction_ohe = self.OHE_4X4[direction]

        distance_to_walls = np.array(
            [head_x, head_y, field_size - 1 - head_x, field_size - 1 - head_y])

        distance_to_apple = np.array((apple_x - head_x, apple_y - head_y))

        frame = self.surface_to_numpy(surface)[::self.__game(
        ).setup.BLOCK_SIZE, ::self.__game().setup.BLOCK_SIZE, 1]

        frame = np.pad(frame, 1, mode="constant",
                       constant_values=Color.SNAKE_COLOR.value[1])
        frame_slice = frame[head_y: head_y + 3, head_x: head_x + 3].flatten()
        frame_slice = np.delete(frame_slice, 4)

        body_part_near = np.array(
            frame_slice == Color.SNAKE_COLOR.value[1], dtype="int8")

        return np.concatenate((direction_ohe, distance_to_walls, distance_to_apple, body_part_near)).astype("int8")


class Game:

    INFO = {
        "actions": {
            "Turn left": -1,
            "Do nothing": 0,
            "Turn right": 1,
        },
        "rewards": {
            "Apple eaten": 10,
            "Went to fruit": 0.01,
            "Went from fruit": -0.01,
            "Died": -10,
        },
        "modes": (
            [
                mode.name for mode
                in
                RenderMode
            ],
        )
    }

    def __init__(self, __field_size: int | tuple[int, int], mode: RenderMode = RenderMode.RGB_IMAGE, verbose: bool = True):
        self.setup = Setup(__field_size)
        self.driver = NumpyDriver(weakref.ref(self))
        self.mode = mode
        self.verbose = verbose

        if self.verbose:
            self.display = pygame.display.set_mode(
                (self.setup.DISPLAY_SIZE, self.setup.DISPLAY_SIZE)
            )
            self.clock = pygame.time.Clock()

        self.surface: Surface
        self.snake: "Snake"
        self.fruit: "Fruit"

        self.done: bool
        self.score: int
        self.fruit_to_snake_distance_previous: int
        self.fruit_to_snake_distance_current: int

    def reset(self):
        self.surface = Surface(
            (self.setup.DISPLAY_SIZE, self.setup.DISPLAY_SIZE))
        self.snake = Snake(self.setup)
        self.fruit = Fruit(self.setup)
        self.fruit.update(*self.__generate_fruit_position())

        self.done = False
        self.score = self.setup.SNAKE_LEN
        self.fruit_to_snake_distance_current = self.__get_distance_between_snake_and_fruit()

        self.__render((*self.snake, self.fruit.block))

        if self.verbose:
            self.__display()

        return self.state, self.INFO

    @property
    def state(self):
        state: np.ndarray

        match self.mode:
            case RenderMode.RGB_IMAGE:
                state = self.driver.surface_to_numpy(
                    self.surface)

            case RenderMode.GRAYSCALE_IMAGE:
                state = self.driver.surface_to_numpy(
                    self.surface)[..., 1]

            case RenderMode.GRAYSCALE_IMAGE_SHRINKED:
                state = self.driver.surface_to_numpy(
                    self.surface)[::self.setup.BLOCK_SIZE, ::self.setup.BLOCK_SIZE, 1]

            case RenderMode.BINARY_VECTOR:
                state = self.driver.generate_binary_array(
                    self.surface, self.snake.direction.value, self.snake.head.coords.xy, self.fruit.block.coords.xy, self.snake.is_dead)

            case RenderMode.DISTANCED_VECTOR:
                state = self.driver.generate_distanced_array(
                    self.surface, self.snake.direction.value, self.snake.head.coords.xy, self.fruit.block.coords.xy, self.snake.is_dead)

        return state

    def step(self, action: int):
        reward: int

        if not self.done:
            self.snake.direction = self.driver.action_to_direction(
                self.snake.direction, action)
            self.snake.move()

            self.fruit_to_snake_distance_previous = self.fruit_to_snake_distance_current
            self.fruit_to_snake_distance_current = self.__get_distance_between_snake_and_fruit()

            if self.snake.is_dead:
                self.done = True
                reward = -10

            elif self.__food_collision():
                self.snake.eat()
                self.fruit.update(*self.__generate_fruit_position())
                self.score += 1
                reward = 10

            else:
                diff = self.fruit_to_snake_distance_current - \
                    self.fruit_to_snake_distance_previous
                reward = np.sign(diff) * -0.01

            self.__render((self.fruit.block, *self.snake))

            if self.verbose:
                self.__display()

        return self.state, reward, self.done, self.INFO

    def __food_collision(self):
        if self.fruit.block.coords == self.snake.head.coords:
            return True
        return False

    def __render(self, blocks: Sequence[Block]):
        self.surface.fill(Color.BLACK.value)
        for block in blocks:
            self.__create_rect(block)

    def __display(self):
        self.__handle_window_events()
        self.display.blit(self.surface, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.setup.GAME_SPEED)

    def __create_rect(self, block: Block):
        coords = block.coords * self.setup.BLOCK_SIZE
        return pygame.draw.rect(self.surface, block.color.value, (*coords, block.size, block.size))

    def __generate_fruit_position(self) -> tuple[int, int]:
        all_positions = set(range(0, self.setup.FIELD_SIZE**2))
        snake_blocks = set(map(lambda block: block.coords.x +
                           block.coords.y * self.setup.FIELD_SIZE, self.snake))
        try:
            block_num = random.choice(tuple(all_positions - snake_blocks))
        except IndexError:
            self.done = True
            self.score -= 1
            block_num = 0

        x = block_num % self.setup.FIELD_SIZE
        y = block_num // self.setup.FIELD_SIZE

        return x, y

    def __handle_window_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def __get_distance_between_snake_and_fruit(self):
        return self.snake.head.coords.distance_to(self.fruit.block.coords)


class Fruit:

    def __init__(self, setup: Setup):
        self.setup = setup
        self.color = Color.FRUIT_COLOR
        self.block: Block

    def update(self, x: int, y: int):
        coords = Vector2(x, y)
        self.block = Block(coords, color=self.color)


class Snake(deque[Block]):

    def __init__(self, setup: Setup):
        super().__init__([])
        self.setup = setup
        self.color = Color.SNAKE_COLOR
        self.head_color = Color.HEAD_COLOR
        self.direction = Direction.RIGHT
        self.__create_snake()

    @property
    def head(self):
        return self[0]

    @property
    def tail(self):
        return self[-1]

    @property
    def is_dead(self):
        return self.__bumped_into_itself() or self.__bumped_into_wall()

    def move(self):
        self.pop()
        coords = self.__get_coords_move()
        block = Block(coords, color=self.head_color)
        self.head.color = self.color
        self.appendleft(block)

    def eat(self):
        tail = deepcopy(self.tail)
        self.append(tail)

    def __get_coords_move(self):
        return self.head.coords + self.direction.mask

    def __create_snake(self):
        x = random.randint(0, self.setup.FIELD_SIZE - 3)
        y = random.randint(0, self.setup.FIELD_SIZE - 1)
        for i in range(self.setup.SNAKE_LEN):
            block_coords = Vector2(i + x, y)
            block = Block(block_coords, color=self.color)
            self.appendleft(block)

        self.head.color = self.head_color

    def __bumped_into_wall(self):
        x, y = self.head.coords

        if x < 0 or x >= self.setup.FIELD_SIZE:
            return True

        if y < 0 or y >= self.setup.FIELD_SIZE:
            return True

        return False

    def __bumped_into_itself(self):
        return len((*filter(lambda block: block.coords == self.head.coords, self), )) != 1


if __name__ == "__main__":
    print(list(filter(lambda x: x.__contains__("jet"), pygame.font.get_fonts())))
    print(Direction.mask)
    game = Game()
