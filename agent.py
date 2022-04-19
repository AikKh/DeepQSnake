import torch
import random
import numpy as np
from collections import deque
from game import Game
from snake import Snake
from model import Linear_QNet, QTrainer


MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
GENERATION = 1


class Agent:
    
    def __init__(self, gen = 0):
        self.n_games = 0
        self.epsilion = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        
        self.model = Linear_QNet(11, 256, 3, gen=True)
        if gen:
            checkpoint = torch.load(f'Generations/gen-{gen}.pth')
            self.model.load_state_dict(checkpoint)
            
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        
                

    def get_state(self, game: Game):
        snake = game.snake
        hx, hy = snake._head
        fx, fy = snake._apple
        
        get_danger = lambda x, y: 0 if 0 <= x < 60 and 0 <= x < 60 and 0 <= y < 60 and 0 <= y < 60 else 1
        
        danger = [get_danger(hx+x, hy+y) for x, y in snake.dirs if (x, y) != snake.dirs[(snake._dir + 2) % 4]]
        dirs = [i == snake.dirs[snake._dir] for i in snake.dirs]
        food = [
            hx > fx,
            hx < fx,
            hy > fy,
            hy < fy,
        ]
        return np.array([*danger, *dirs, *food], dtype=int)
    
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
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
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
    global GENERATION
    record = 0
    
    agent = Agent()
    game = Game()
    
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        reward, done, score = game.snake.step(final_move)
        state_new = agent.get_state(game)
        
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        
        game.blit(f'Generation: {agent.n_games}  Score: {score}  Record: {record}')
        game.draw()
        
        if done:
            game.snake = Snake()
            agent.n_games += 1
            agent.train_long_memory()
            
            if score > record:
                record = score
                agent.model.save(GENERATION)

            # print('Game', agent.n_games, 'Score', score, 'Record:', record)
            GENERATION += 1
            
            
def test(gen=0):
    game = Game()
    
    if gen:
        agent = Agent(gen)
    else:
        agent = Agent()
    
    while not game.snake._die:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        
        game.snake.step(final_move)
        
        game.blit(f'Generation: {gen}')
        game.draw()
        


if __name__ == '__main__':
    # train()
    for i in [5, 24, 42, 67, 72, 81, 82, 86, 90, 95]:
        test(i)
    
    