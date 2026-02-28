import pygame
import sys

# Configuration
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
CELL_SIZE = 40
GRID_COLOR = (0, 100, 0)        # Dark green grid lines
BACKGROUND_COLOR = (0, 200, 80) # Bright green background
SQUARE_COLOR = (255, 255, 255)  # White square
LINE_WIDTH = 1

# How many frames between each automatic step
MOVE_INTERVAL = 10  # lower = faster

# Grid dimensions (in cells)
COLS = WINDOW_WIDTH // CELL_SIZE
ROWS = WINDOW_HEIGHT // CELL_SIZE


def check_barrier(grid_x, grid_y, dx, dy):
    """
    Check whether moving by (dx, dy) grid steps would take the square
    outside the screen boundaries.
    Returns True if the move is allowed, False if it is blocked.
    """
    new_x = grid_x + dx
    new_y = grid_y + dy

    if new_x < 0 or new_x >= COLS:
        return False  # Blocked: left or right wall
    if new_y < 0 or new_y >= ROWS:
        return False  # Blocked: top or bottom wall
    return True       # Move is safe


def movement(grid_x, grid_y, direction):
    """
    Move the square one cell in the current direction automatically.
    direction is a (dx, dy) tuple set by the last WASD key press.
    If the move is blocked by a barrier the square simply stops.
    Returns the new grid position.
    """
    dx, dy = direction

    if dx == 0 and dy == 0:
        return grid_x, grid_y  # No direction set yet

    if check_barrier(grid_x, grid_y, dx, dy):
        return grid_x + dx, grid_y + dy
    else:
        return grid_x, grid_y  # Blocked — stay in place


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Green Grid — WASD to steer")
    clock = pygame.time.Clock()

    # Square starts in the middle of the grid
    square_x = COLS // 2
    square_y = ROWS // 2

    # Current direction as (dx, dy); starts stationary
    direction = (0, 0)

    # Frame counter to throttle automatic movement speed
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
                # Player only sets the direction; movement happens automatically
                elif event.key == pygame.K_w:
                    direction = (0, -1)
                elif event.key == pygame.K_s:
                    direction = (0, 1)
                elif event.key == pygame.K_a:
                    direction = (-1, 0)
                elif event.key == pygame.K_d:
                    direction = (1, 0)

        # Advance the square one cell every MOVE_INTERVAL frames
        frame_count += 1
        if frame_count >= MOVE_INTERVAL:
            frame_count = 0
            square_x, square_y = movement(square_x, square_y, direction)

        # --- Drawing ---
        screen.fill(BACKGROUND_COLOR)

        # Draw vertical grid lines
        for x in range(0, WINDOW_WIDTH + 1, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, WINDOW_HEIGHT), LINE_WIDTH)

        # Draw horizontal grid lines
        for y in range(0, WINDOW_HEIGHT + 1, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WINDOW_WIDTH, y), LINE_WIDTH)

        # Draw the player square (slightly inset so grid lines stay visible)
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