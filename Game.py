import pygame
import sys


WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 40
GRID_COLOR = (0, 100, 0)        
BACKGROUND_COLOR = (0, 200, 80) 
SQUARE_COLOR = (255, 255, 255)  
LINE_WIDTH = 1


MOVE_INTERVAL = 10  


COLS = WINDOW_WIDTH // CELL_SIZE
ROWS = WINDOW_HEIGHT // CELL_SIZE


def check_barrier(grid_x, grid_y, dx, dy):
    
    new_x = grid_x + dx
    new_y = grid_y + dy

    if new_x < 0 or new_x >= COLS:
        return False  
    if new_y < 0 or new_y >= ROWS:
        return False  
    return True       


def movement(grid_x, grid_y, direction):
    
    dx, dy = direction

    if dx == 0 and dy == 0:
        return grid_x, grid_y  

    if check_barrier(grid_x, grid_y, dx, dy):
        return grid_x + dx, grid_y + dy
    else:
        return grid_x, grid_y  


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Green Grid — WASD to steer")
    clock = pygame.time.Clock()

    
    square_x = COLS // 2
    square_y = ROWS // 2

   
    direction = (0, 0)

    
    frame_count = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                
                elif event.key == pygame.K_w:
                    direction = (0, -1)
                elif event.key == pygame.K_s:
                    direction = (0, 1)
                elif event.key == pygame.K_a:
                    direction = (-1, 0)
                elif event.key == pygame.K_d:
                    direction = (1, 0)

       
        frame_count += 1
        if frame_count >= MOVE_INTERVAL:
            frame_count = 0
            square_x, square_y = movement(square_x, square_y, direction)

      
        screen.fill(BACKGROUND_COLOR)

        
        for x in range(0, WINDOW_WIDTH + 1, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT), LINE_WIDTH)

        
        for y in range(0, WINDOW_HEIGHT + 1, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WINDOW_WIDTH, y), LINE_WIDTH)

        
        padding = 4
        rect = pygame.Rect(
            square_x * CELL_SIZE + padding,
            square_y * CELL_SIZE + padding,
            CELL_SIZE - padding * 2,
            CELL_SIZE - padding * 2
        )
        pygame.draw.rect(screen, SQUARE_COLOR, rect, border_radius=4)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()