import os
import pickle
import random
import pygame
from module_new_world import World, TerrainCell

# ───────────────────── Configuration ─────────────────────

GRID_SIZE      = 40        # number of cells per row/column
CELL_SIZE      = 16        # pixels per cell
SIDEBAR_WIDTH  = 200       # width of the UI panel
WORLD_PIXELS   = GRID_SIZE * CELL_SIZE
WINDOW_WIDTH   = WORLD_PIXELS + SIDEBAR_WIDTH
WINDOW_HEIGHT  = WORLD_PIXELS
FPS            = 30
SAVE_PATH      = "world.pkl"

# ───────────────────── Materials & Tools ─────────────────────

MATERIAL_COLORS = {
    "dirt":    (210,180,140),
    "stone":   (169,169,169),
    "rock":    (128,128,128),
    "water":   ( 30,144,255),
    "wood":    ( 34,139, 34),
    "food":    (220, 20, 60),
    "home":    (255,215,  0),
}

TOOLS = ["boulder", "ridge", "river", "forest", "dirt", "home", "food"]
TOOL_MATERIAL = {
    "boulder": "stone",
    "ridge":   "rock",
    "river":   "water",
    "forest":  "wood",
    "dirt":    "dirt",
    "home":    "home",
    "food":    "food",
}

def create_cell(tool, world):
    ambient = world.ambient_temperature
    if tool == "boulder":
        return TerrainCell(
            height_vector=(0,0), normal_vector=(0,0),
            material="stone", passable=True,
            hardness=7.0, strength=8.0, density=2.5,
            friction=1.2, elasticity=0.0,
            thermal_conductivity=0.3,
            temperature=ambient,
            local_risk=0.05,
            tags={"boulder"}
        )
    elif tool == "ridge":
        return TerrainCell(
            height_vector=(0,0), normal_vector=(0,0),
            material="rock", passable=True,
            hardness=9.0, strength=10.0, density=2.7,
            friction=1.0, elasticity=0.0,
            thermal_conductivity=0.2,
            temperature=ambient,
            local_risk=0.0,
            tags={"cliff","ridge"}
        )
    elif tool == "river":
        return TerrainCell(
            height_vector=(0,0), normal_vector=(0,0),
            material="water", passable=True,
            hardness=0.5, strength=0.5, density=1.0,
            friction=2.0, elasticity=0.0,
            thermal_conductivity=0.15,
            temperature=ambient,
            local_risk=0.2,
            tags={"river"}
        )
    elif tool == "forest":
        return TerrainCell(
            height_vector=(0,0), normal_vector=(0,0),
            material="wood", passable=True,
            hardness=2.0, strength=4.0, density=0.9,
            friction=1.5, elasticity=0.0,
            thermal_conductivity=0.25,
            temperature=ambient,
            local_risk=0.0,
            tags={"forest","tree"}
        )
    elif tool == "home":
        ox, oy = world.home
        world.grid[ox][oy] = create_cell("dirt", world)
        return TerrainCell(
            height_vector=(0,0), normal_vector=(0,0),
            material="home", passable=True,
            hardness=1.0, strength=1.0, density=1.0,
            friction=1.0, elasticity=0.0,
            thermal_conductivity=0.1,
            temperature=ambient,
            local_risk=0.0,
            tags={"home"}
        )
    elif tool == "food":
        return TerrainCell(
            height_vector=(0,0), normal_vector=(0,0),
            material="food", passable=True,
            hardness=0.5, strength=0.5, density=1.0,
            friction=1.0, elasticity=0.0,
            thermal_conductivity=0.1,
            temperature=ambient,
            local_risk=0.0,
            tags={"food"}
        )
    else:  # dirt
        return TerrainCell(
            height_vector=(0,0), normal_vector=(0,0),
            material="dirt", passable=True,
            hardness=1.0, strength=1.0, density=1.0,
            friction=1.0, elasticity=0.0,
            thermal_conductivity=0.1,
            temperature=ambient,
            local_risk=0.0,
            tags=set()
        )

def draw_world(screen, world):
    for x in range(world.grid_size):
        for y in range(world.grid_size):
            cell = world.cell((x, y))
            color = MATERIAL_COLORS.get(cell.material, (240,240,240))
            rect = pygame.Rect(y*CELL_SIZE, x*CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, (200,200,200), rect, 1)

def draw_ui(screen, tool_positions, random_rect, action_rects, current_tool, fonts):
    header_font, item_font = fonts
    # Sidebar background
    pygame.draw.rect(screen, (40,40,40), (WORLD_PIXELS, 0, SIDEBAR_WIDTH, WINDOW_HEIGHT))
    # Header
    header_surf = header_font.render("Tools & Controls", True, (255,255,255))
    screen.blit(header_surf, (WORLD_PIXELS + 10, 10))

    # Tool buttons
    for tool, rect in tool_positions.items():
        active = (tool == current_tool)
        color = (255,215,0) if active else (200,200,200)
        mat = TOOL_MATERIAL.get(tool)
        swatch = MATERIAL_COLORS.get(mat, (255,255,255))
        pygame.draw.circle(screen, swatch, (rect.x + 10, rect.y + rect.h//2), rect.h//3)
        text = item_font.render(tool.title(), True, color)
        screen.blit(text, (rect.x + 30, rect.y))

    # Random world button
    pygame.draw.rect(screen, (70,70,70), random_rect)
    rnd_surf = item_font.render("Generate Random", True, (255,255,255))
    screen.blit(rnd_surf, (random_rect.x+10, random_rect.y+5))

    # Action buttons: Save, Load, New, Quit
    for name, rect in action_rects.items():
        pygame.draw.rect(screen, (70,70,70), rect)
        txt = item_font.render(name.title(), True, (255,255,255))
        screen.blit(txt, (rect.x+10, rect.y+5))

def save_world(world, path=SAVE_PATH):
    with open(path, "wb") as f:
        pickle.dump(world, f)

def load_world(path=SAVE_PATH):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return World(GRID_SIZE)

def random_build(world):
    world.__init__(world.grid_size)
    # Ridges
    for _ in range(random.randint(2,4)):
        ori = random.choice(['h','v'])
        if ori=='h':
            start=(random.randint(0,world.grid_size-1),0); dir=(1,0)
        else:
            start=(0,random.randint(0,world.grid_size-1)); dir=(0,1)
        length=random.randint(world.grid_size//2, world.grid_size)
        world.add_ridge(start, length, dir)
    # Curved Rivers
    for _ in range(random.randint(1,3)):
        edge=random.choice(['top','bottom','left','right'])
        if edge=='top':
            x,y=0,random.randint(0,world.grid_size-1); dx,dy=1,0
        elif edge=='bottom':
            x,y=world.grid_size-1,random.randint(0,world.grid_size-1); dx,dy=-1,0
        elif edge=='left':
            x,y=random.randint(0,world.grid_size-1),0; dx,dy=0,1
        else:
            x,y=random.randint(0,world.grid_size-1),world.grid_size-1; dx,dy=0,-1
        length=random.randint(world.grid_size//2, world.grid_size)
        for _ in range(length):
            if not (0<=x<world.grid_size and 0<=y<world.grid_size): break
            world.grid[x][y] = create_cell("river", world)
            if random.random()<0.3:
                if random.random()<0.5: dx,dy = -dy, dx
                else:            dx,dy = dy, -dx
            x,y = x+dx, y+dy
    # Forests
    for _ in range(random.randint(3,6)):
        c= (random.randint(0,world.grid_size-1), random.randint(0,world.grid_size-1))
        r= random.randint(2,5)
        world.add_forest(c,r)
    # Boulders
    for _ in range(random.randint(5,10)):
        c= (random.randint(0,world.grid_size-1), random.randint(0,world.grid_size-1))
        r= random.randint(1,3)
        world.add_boulder(c,r)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("🗺️ World Editor")
    clock = pygame.time.Clock()

    # Fonts
    header_font = pygame.font.SysFont(None, 28)
    item_font   = pygame.font.SysFont(None, 22)
    fonts = (header_font, item_font)

    # Sidebar layout
    tool_positions = {}
    y = 50; lh = item_font.get_height() + 12
    for tool in TOOLS:
        tool_positions[tool] = pygame.Rect(WORLD_PIXELS+10, y, SIDEBAR_WIDTH-20, item_font.get_height())
        y += lh

    random_rect = pygame.Rect(WORLD_PIXELS+10, y+20, SIDEBAR_WIDTH-20, item_font.get_height()+8)
    action_rects = {}
    ay = random_rect.y + random_rect.h + 20
    for name in ["save", "load", "new", "quit"]:
        action_rects[name] = pygame.Rect(WORLD_PIXELS+10, ay, SIDEBAR_WIDTH-20, item_font.get_height()+8)
        ay += item_font.get_height()+12

    world = World(GRID_SIZE)
    current_tool = TOOLS[0]
    dragging = False

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                mx,my = ev.pos
                if mx >= WORLD_PIXELS:
                    # Sidebar clicks
                    for tool, rect in tool_positions.items():
                        if rect.collidepoint(ev.pos):
                            current_tool = tool
                            break
                    if random_rect.collidepoint(ev.pos):
                        random_build(world)
                    for name, rect in action_rects.items():
                        if rect.collidepoint(ev.pos):
                            if name=="save":
                                save_world(world)
                            elif name=="load":
                                world = load_world()
                            elif name=="new":
                                world = World(GRID_SIZE)
                            elif name=="quit":
                                running = False
                            break
                else:
                    dragging = True
                    gx,gy = ev.pos[1]//CELL_SIZE, ev.pos[0]//CELL_SIZE
                    if current_tool=="home":
                        world.home = (gx,gy)
                    world.grid[gx][gy] = create_cell(current_tool, world)

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                dragging = False

            elif ev.type == pygame.MOUSEMOTION and dragging:
                mx,my = ev.pos
                if mx < WORLD_PIXELS:
                    gx,gy = my//CELL_SIZE, mx//CELL_SIZE
                    if current_tool=="home":
                        world.home = (gx,gy)
                    world.grid[gx][gy] = create_cell(current_tool, world)

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_s:
                    save_world(world)
                elif ev.key == pygame.K_l:
                    world = load_world()
                elif ev.key == pygame.K_n:
                    world = World(GRID_SIZE)
                elif ev.key == pygame.K_q:
                    running = False

        screen.fill((0,0,0))
        draw_world(screen, world)
        draw_ui(screen, tool_positions, random_rect, action_rects, current_tool, fonts)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
