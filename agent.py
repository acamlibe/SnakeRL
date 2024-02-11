import torch
import random
import numpy as np

from collections import deque
from game import SnakeGameAI, Direction, Point, BLOCK_SIZE
from model import Linear_QNet, QTrainer
from helper import plot

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.num_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        # TODO: model, trainer
        self.model = Linear_QNet(input_size=11, hidden_size=256, output_size=3)
        self.trainer = QTrainer(self.model, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        point_left = Point(head.x - BLOCK_SIZE, head.y)
        point_right = Point(head.x + BLOCK_SIZE, head.y)
        point_up = Point(head.x, head.y - BLOCK_SIZE)
        point_down = Point(head.x, head.y + BLOCK_SIZE)

        dir_left = game.direction == Direction.LEFT
        dir_right = game.direction == Direction.RIGHT
        dir_up = game.direction == Direction.UP
        dir_down = game.direction == Direction.DOWN

        state = [
            # Danger Straight
            (dir_left and game.is_collision(point_left)) or
            (dir_right and game.is_collision(point_right)) or
            (dir_up and game.is_collision(point_up)) or
            (dir_down and game.is_collision(point_down)),

            # Danger Right
            (dir_left and game.is_collision(point_up)) or
            (dir_right and game.is_collision(point_down)) or
            (dir_up and game.is_collision(point_right)) or
            (dir_down and game.is_collision(point_up)),

            # Danger Left
            (dir_left and game.is_collision(point_down)) or
            (dir_right and game.is_collision(point_up)) or
            (dir_up and game.is_collision(point_left)) or
            (dir_down and game.is_collision(point_right)),

            # Move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,

            # Food Location
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]
    
        return np.array(state, dtype=int)
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # Random moves: tradeoff between exploration and exploitation
        self.epsilon = 80 - self.num_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    agent = Agent()
    game = SnakeGameAI()

    while True:
        state = agent.get_state()
        move = agent.get_action(state)

        reward, done, score = game.play_step(move)
        new_state = agent.get_state(game)

        agent.train_short_memory(state, move, reward, new_state, done)
        agent.remember(state, move, reward, new_state, done)

        if done:
            game.reset()
            agent.num_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game:', agent.num_games, 'Score:', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.num_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        


if __name__ == '__name__':
    train()