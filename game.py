import pygame, sys
from snake import Snake


WHITE = 255, 255, 255
BLACK = 40, 40, 40
RED = 255, 0, 0


class Game:
    
    pygame.init()
    width = height = 600
    screen = pygame.display.set_mode((width, height))
    
    pygame.display.set_caption("SnakeAI")

    clock = pygame.time.Clock()
    fps = 50
    
    font = pygame.font.Font('freesansbold.ttf', 20)
    
    
    def __init__(self):
        self.snake = Snake()
        
    def blit(self, txt: str):
        text = self.font.render(txt, True, WHITE)
        self.screen.blit(text, (20, 20))
        
    def draw(self):
        pix = 10
        pygame.draw.rect(self.screen, RED, (self.snake._apple[0]*pix, self.snake._apple[1]*pix, pix, pix))
        
        for x, y in self.snake._cors:
            pygame.draw.rect(self.screen, self.snake._color, (x*pix, y*pix, pix, pix))
            
        pygame.display.flip()
        self.clock.tick(self.fps)
        self.screen.fill(BLACK)
