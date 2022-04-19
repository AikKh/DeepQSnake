import random

GREEN = 0, 255, 0


class Snake:
    
    dirs = [(1, 0), (0, 1), (-1, 0) , (0, -1)]
    
    def __init__(self,):
        
        self._dir = 0
        self._cors = [(30, 30), (29, 30)]
        
        self._head = self._cors[0]
        self._size = len(self._cors)
        self._color = GREEN
        self._apple = self.get_apple
        
        self._die = False
        
    
    @property
    def get_apple(self):
        free_space = [(x, y) for x in range(60) for y in range(60) if (x, y) not in self._cors]
        return random.choice(free_space)
        
        
    def move(self, action):
        # print(action, type(action))
        self._dir += {tuple([1, 0, 0]): 0,
                      tuple([0, 1, 0]): 1,
                      tuple([0, 0, 1]): -1}[tuple(action)]
        
        self._dir %= 4
        x, y = self.dirs[self._dir]
        
        old_one = self._cors[0]
        self._cors[0] = (self._cors[0][0] + x, self._cors[0][1] + y)
        for i in range(1, self._size):
            self._cors[i], old_one = old_one, self._cors[i]
        
        self._head = self._cors[0]
        
    def eat(self):
        x, y = self.dirs[self._dir]
        self._cors.append((self._cors[-1][0] + x, self._cors[-1][0] + y))
        self._size += 1
        self.move_count = 0
        self._apple = self.get_apple
        
        
    def step(self, action):
        self.move(action)
        
        hx, hy = self._head
        reward = 0
        
        if not (0 <= hx < 60 and 0 <= hy < 60) or (hx, hy) in self._cors[1:]:
            self._die = True
            reward = -10
            return reward, self._die, self._size - 2 
            
        elif (hx, hy) == self._apple:
            reward = 10
            self.eat()
            
        return reward, self._die, self._size - 2