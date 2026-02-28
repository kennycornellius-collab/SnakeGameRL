import pygame
import sys
import random

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 40
GRID_COLOR = (0, 100, 0)        
BACKGROUND_COLOR = (0, 200, 80) 
SQUARE_COLOR = (255, 255, 255)  
FOOD_COLOR = (220, 30, 30)      
LINE_WIDTH = 1


MOVE_INTERVAL = 10  


COLS = WINDOW_WIDTH // CELL_SIZE
ROWS = WINDOW_HEIGHT // CELL_SIZE


def check_barrier(head_x, head_y, dx, dy, body):
    
    new_x = head_x + dx
    new_y = head_y + dy

    if new_x < 0 or new_x >= COLS or new_y < 0 or new_y >= ROWS:
        pygame.quit()

    if (new_x, new_y) in set(body):
        pygame.quit()


def movement(body, direction, grow):
    
    dx, dy = direction

    if dx == 0 and dy == 0:
        return body, False  

    head_x, head_y = body[0]

    check_barrier(head_x, head_y, dx, dy, body)

    new_head = (head_x + dx, head_y + dy)
    new_body = [new_head] + body[:-1] if not grow else [new_head] + body
    return new_body, True


def spawn_food(body):
    
    occupied = set(body)
    while True:
        pos = (random.randint(0, COLS - 1), random.randint(0, ROWS - 1))
        if pos not in occupied:
            return pos


def draw_cell(screen, x, y, color, padding=4):
    rect = pygame.Rect(
        x * CELL_SIZE + padding,
        y * CELL_SIZE + padding,
        CELL_SIZE - padding * 2,
        CELL_SIZE - padding * 2
    )
    pygame.draw.rect(screen, color, rect, border_radius=4)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Green Grid — WASD to steer")
    clock = pygame.time.Clock()

   
    body = [(COLS // 2, ROWS // 2)]
    direction = (0, 0)

    
    food = spawn_food(body)

    
    grow = False

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
            body, moved = movement(body, direction, grow)
            grow = False  

           
            if moved and body[0] == food:
                grow = True              
                food = spawn_food(body)  

        
        screen.fill(BACKGROUND_COLOR)

        
        for x in range(0, WINDOW_WIDTH + 1, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT), LINE_WIDTH)

       
        for y in range(0, WINDOW_HEIGHT + 1, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WINDOW_WIDTH, y), LINE_WIDTH)

        
        draw_cell(screen, food[0], food[1], FOOD_COLOR)

        
        for i, (bx, by) in enumerate(body):
            color = SQUARE_COLOR if i == 0 else (180, 180, 180)
            draw_cell(screen, bx, by, color)

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()