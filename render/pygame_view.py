import pygame

CELL = 70
COLORS = {
    0: (245, 240, 230),
    1: (70, 70, 90),
    2: (80, 150, 230),
    3: (130, 80, 40),
}
ROBOT = (60, 180, 120)
BACKGROUND = (30, 30, 40)
BAR_HEIGHT = 40


class PygameView:
    def __init__(self, env, title="Coffee Bot"):
        self.env = env
        pygame.init()
        width = env.cols * CELL
        height = env.rows * CELL + BAR_HEIGHT
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        self.font = pygame.font.Font(None, 28)
        self.clock = pygame.time.Clock()

    def draw(self, label=""):
        self.screen.fill(BACKGROUND)
        for r in range(self.env.rows):
            for c in range(self.env.cols):
                rect = (c * CELL, r * CELL, CELL - 2, CELL - 2)
                pygame.draw.rect(self.screen, COLORS[int(self.env.grid[r, c])], rect)

        for r in range(self.env.rows):
            for c in range(self.env.cols):
                if int(self.env.grid[r, c]) == 3:
                    x, y = c * CELL, r * CELL
                    pygame.draw.rect(self.screen, (250, 250, 250), (x + 18, y + 24, 30, 28))
                    pygame.draw.circle(self.screen, (250, 250, 250), (x + 52, y + 38), 9, 4)
                    pygame.draw.rect(self.screen, (90, 50, 20), (x + 18, y + 24, 30, 8))

        pr, pc = self.env.pos
        center = (pc * CELL + CELL // 2, pr * CELL + CELL // 2)
        pygame.draw.circle(self.screen, ROBOT, center, CELL // 3)

        text = self.font.render(label, True, (240, 240, 240))
        self.screen.blit(text, (10, self.env.rows * CELL + 10))
        pygame.display.flip()

    def tick(self, fps=4):
        self.clock.tick(fps)

    def should_quit(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
        return False

    def close(self):
        pygame.quit()