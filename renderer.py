import sys
import pygame

CELL_SIZE       = 40
BACKGROUND      = (0, 200, 80)
GRID_COLOR      = (0, 100, 0)
HEAD_COLOR      = (255, 255, 255)
BODY_COLOR      = (180, 180, 180)
FOOD_COLOR      = (220, 30, 30)
LINE_WIDTH      = 1
CELL_PADDING    = 4


class Renderer:
    def __init__(self, cols: int, rows: int, fps: int = 10):
        self.cols = cols
        self.rows = rows
        self.fps  = fps

        pygame.init()
        w = cols * CELL_SIZE
        h = rows * CELL_SIZE
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("Snake — RL")
        self.clock = pygame.time.Clock()
        self._w = w
        self._h = h

    def draw(self, state: dict):
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.close()
                sys.exit()

        self.screen.fill(BACKGROUND)
        self._draw_grid()
        self._draw_cell(*state["food"], FOOD_COLOR)
        for i, (bx, by) in enumerate(state["body"]):
            color = HEAD_COLOR if i == 0 else BODY_COLOR
            self._draw_cell(bx, by, color)

        score_surf = pygame.font.SysFont(None, 28).render(
            f"Score: {state['score']}", True, (0, 60, 0)
        )
        self.screen.blit(score_surf, (6, 4))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()


    def _draw_grid(self):
        for x in range(0, self._w + 1, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (x, 0), (x, self._h), LINE_WIDTH)
        for y in range(0, self._h + 1, CELL_SIZE):
            pygame.draw.line(self.screen, GRID_COLOR, (0, y), (self._w, y), LINE_WIDTH)

    def _draw_cell(self, x: int, y: int, color):
        p = CELL_PADDING
        rect = pygame.Rect(
            x * CELL_SIZE + p,
            y * CELL_SIZE + p,
            CELL_SIZE - p * 2,
            CELL_SIZE - p * 2,
        )
        pygame.draw.rect(self.screen, color, rect, border_radius=4)
