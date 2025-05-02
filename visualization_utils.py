import pygame
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# ─────────────────── Visualization Utilities ───────────────────

def create_graph_surface(agent, width=350, height=200):
    """Create a simplified surface with agent history visualization"""
    # Create a new surface
    surface = pygame.Surface((width, height))
    surface.fill((50, 50, 50))  # Dark gray background
    
    if len(agent.history["energy"]) < 10:
        # Not enough data for a meaningful graph
        font = pygame.font.SysFont("Arial", 14)
        text = font.render("Not enough data for visualization", True, (200, 200, 200))
        surface.blit(text, (width//2 - text.get_width()//2, height//2 - text.get_height()//2))
        return surface
    
    # Get the most recent data (last 300 points)
    max_points = 300
    data_points = min(len(agent.history["energy"]), max_points)
    
    # Get the data
    energy = agent.history["energy"][-data_points:]
    hunger = agent.history["hunger"][-data_points:]
    pain = agent.history["pain"][-data_points:]
    
    # Determine the range of values
    all_values = energy + hunger + pain
    if "rewards" in agent.history and len(agent.history["rewards"]) >= data_points:
        rewards = agent.history["rewards"][-data_points:]
        all_values += rewards
    
    min_val = min(all_values) if all_values else 0
    max_val = max(all_values) if all_values else 100
    
    # Make room for the negative rewards
    if min_val < 0:
        min_val = min(min_val, -10)  # At least -10
    
    # Add some padding to the range
    min_val -= 5
    max_val += 5
    
    # Calculate the scale factors
    x_scale = width / data_points
    if max_val > min_val:
        y_scale = height / (max_val - min_val)
    else:
        y_scale = 1.0
    
    # Draw grid lines
    for i in range(0, height, 20):
        pygame.draw.line(surface, (70, 70, 70), (0, i), (width, i), 1)
    
    for i in range(0, width, 20):
        pygame.draw.line(surface, (70, 70, 70), (i, 0), (i, height), 1)
    
    # Draw zero line if min_val is negative
    if min_val < 0:
        zero_y = height - (0 - min_val) * y_scale
        pygame.draw.line(surface, (100, 100, 100), (0, zero_y), (width, zero_y), 2)
    
    # Draw the data lines
    def draw_line(data, color):
        points = []
        for i, value in enumerate(data):
            x = i * x_scale
            y = height - (value - min_val) * y_scale
            points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(surface, color, False, points, 2)
    
    # Draw lines for each metric
    draw_line(energy, (50, 150, 255))  # Blue
    draw_line(hunger, (255, 150, 50))  # Orange
    draw_line(pain, (255, 50, 50))    # Red
    
    if "rewards" in agent.history and len(agent.history["rewards"]) >= data_points:
        draw_line(rewards, (50, 255, 50))  # Green
    
    # Draw legend
    legend_y = 10
    legend_spacing = 20
    font = pygame.font.SysFont("Arial", 12)
    
    # Energy
    pygame.draw.line(surface, (50, 150, 255), (10, legend_y), (30, legend_y), 2)
    text = font.render("Energy", True, (200, 200, 200))
    surface.blit(text, (35, legend_y - 6))
    
    # Hunger
    pygame.draw.line(surface, (255, 150, 50), (10, legend_y + legend_spacing), (30, legend_y + legend_spacing), 2)
    text = font.render("Hunger", True, (200, 200, 200))
    surface.blit(text, (35, legend_y + legend_spacing - 6))
    
    # Pain
    pygame.draw.line(surface, (255, 50, 50), (10, legend_y + 2 * legend_spacing), (30, legend_y + 2 * legend_spacing), 2)
    text = font.render("Pain", True, (200, 200, 200))
    surface.blit(text, (35, legend_y + 2 * legend_spacing - 6))
    
    # Rewards (if available)
    if "rewards" in agent.history and len(agent.history["rewards"]) >= data_points:
        pygame.draw.line(surface, (50, 255, 50), (10, legend_y + 3 * legend_spacing), (30, legend_y + 3 * legend_spacing), 2)
        text = font.render("Reward", True, (200, 200, 200))
        surface.blit(text, (35, legend_y + 3 * legend_spacing - 6))
    
    # Draw title
    title_font = pygame.font.SysFont("Arial", 14, bold=True)
    title = title_font.render(f"Agent History (Last {data_points} Ticks)", True, (220, 220, 220))
    surface.blit(title, (width//2 - title.get_width()//2, 10))
    
    # Add border
    pygame.draw.rect(surface, (100, 100, 100), (0, 0, width, height), 1)
    
    return surface

def create_population_stats_surface(population, width=350, height=200):
    """Create a simplified surface with population statistics visualization"""
    # Create a new surface
    surface = pygame.Surface((width, height))
    surface.fill((50, 50, 50))  # Dark gray background
    
    if len(population.agents) < 2:
        # Not enough agents for meaningful statistics
        font = pygame.font.SysFont("Arial", 14)
        text = font.render("Not enough agents for visualization", True, (200, 200, 200))
        surface.blit(text, (width//2 - text.get_width()//2, height//2 - text.get_height()//2))
        return surface
    
    # Calculate agent stats
    energy_values = [agent.energy for agent in population.agents.values()]
    hunger_values = [agent.hunger for agent in population.agents.values()]
    pain_values = [agent.pain for agent in population.agents.values()]
    food_values = [agent.store for agent in population.agents.values()]
    
    # Calculate averages
    avg_energy = sum(energy_values) / len(energy_values)
    avg_hunger = sum(hunger_values) / len(hunger_values)
    avg_pain = sum(pain_values) / len(pain_values)
    avg_food = sum(food_values) / len(food_values)
    
    # Calculate standard deviations
    std_energy = np.std(energy_values) if len(energy_values) > 1 else 0
    std_hunger = np.std(hunger_values) if len(hunger_values) > 1 else 0
    std_pain = np.std(pain_values) if len(pain_values) > 1 else 0
    std_food = np.std(food_values) if len(food_values) > 1 else 0
    
    # Find maximum value for scaling
    max_value = max(avg_energy + std_energy, avg_hunger + std_hunger, avg_pain + std_pain, avg_food + std_food, 100)
    
    # Define bar properties
    bar_count = 4
    bar_width = width / (bar_count * 2)
    bar_spacing = width / (bar_count + 1)
    bar_colors = [(50, 150, 255), (255, 150, 50), (255, 50, 50), (50, 200, 50)]  # Blue, Orange, Red, Green
    bar_labels = ["Energy", "Hunger", "Pain", "Food"]
    bar_values = [avg_energy, avg_hunger, avg_pain, avg_food]
    std_values = [std_energy, std_hunger, std_pain, std_food]
    
    # Draw grid lines
    for i in range(0, height, 20):
        pygame.draw.line(surface, (70, 70, 70), (0, i), (width, i), 1)
    
    # Draw the bars
    font = pygame.font.SysFont("Arial", 12)
    for i in range(bar_count):
        # Calculate bar position and height
        x = bar_spacing * (i + 0.5)
        value_height = (bar_values[i] / max_value) * (height - 40)
        bar_rect = pygame.Rect(x - bar_width/2, height - 30 - value_height, bar_width, value_height)
        
        # Draw the bar
        pygame.draw.rect(surface, bar_colors[i], bar_rect)
        pygame.draw.rect(surface, (150, 150, 150), bar_rect, 1)  # Border
        
        # Draw standard deviation error bars
        if std_values[i] > 0:
            std_height = (std_values[i] / max_value) * (height - 40)
            std_top = max(20, height - 30 - value_height - std_height)
            std_bottom = min(height - 30, height - 30 - value_height + std_height)
            
            # Error bar lines
            pygame.draw.line(surface, (200, 200, 200), 
                           (x, std_top), 
                           (x, std_bottom), 2)
            
            # Error bar caps
            pygame.draw.line(surface, (200, 200, 200), 
                           (x - 5, std_top), 
                           (x + 5, std_top), 2)
            pygame.draw.line(surface, (200, 200, 200), 
                           (x - 5, std_bottom), 
                           (x + 5, std_bottom), 2)
        
        # Draw the label
        label = font.render(bar_labels[i], True, (200, 200, 200))
        surface.blit(label, (x - label.get_width()/2, height - 20))
        
        # Draw the value
        value_text = font.render(f"{bar_values[i]:.1f}", True, (200, 200, 200))
        surface.blit(value_text, (x - value_text.get_width()/2, height - 30 - value_height - 15))
    
    # Draw title
    title_font = pygame.font.SysFont("Arial", 14, bold=True)
    title = title_font.render(f"Population Averages (N={len(population.agents)})", True, (220, 220, 220))
    surface.blit(title, (width//2 - title.get_width()//2, 10))
    
    # Add border
    pygame.draw.rect(surface, (100, 100, 100), (0, 0, width, height), 1)
    
    return surface

def draw_minimap(screen, world, size=150, position=(20, 20)):
    """Draw a small overview map in the corner of the screen"""
    grid_size = world.grid_size
    cell_size = size / grid_size
    
    # Create surface for minimap
    minimap = pygame.Surface((size, size))
    minimap.fill((230, 230, 230))
    
    # Draw cells
    for i in range(grid_size):
        for j in range(grid_size):
            cell = world.cell((i, j))
            
            # Determine color based on material
            if cell.material == "food" or "food" in cell.tags:
                color = (220, 20, 60)  # Red for food
            elif cell.material == "home" or "home" in cell.tags:
                color = (255, 215, 0)  # Gold for home
            elif not cell.passable:
                color = (80, 80, 80)  # Dark gray for impassable
            elif cell.material == "water":
                color = (30, 144, 255)  # Blue for water
            elif cell.material == "wood":
                color = (34, 139, 34)  # Green for wood/forest
            elif cell.material == "stone" or cell.material == "rock":
                color = (169, 169, 169)  # Gray for stone/rock
            else:
                color = (210, 180, 140)  # Brown for dirt/default
            
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(minimap, color, rect)
    
    # Add border
    pygame.draw.rect(minimap, (0, 0, 0), pygame.Rect(0, 0, size, size), 2)
    
    # If screen is provided, blit to it
    if screen is not None:
        screen.blit(minimap, position)
    
    return minimap

def create_interaction_graph(population, width=350, height=200):
    """Create a simplified visualization of recent agent interactions"""
    # Create a new surface
    surface = pygame.Surface((width, height))
    surface.fill((50, 50, 50))  # Dark gray background
    
    if len(population.interactions) < 3:
        # Not enough interactions for meaningful visualization
        font = pygame.font.SysFont("Arial", 14)
        text = font.render("Not enough interactions for visualization", True, (200, 200, 200))
        surface.blit(text, (width//2 - text.get_width()//2, height//2 - text.get_height()//2))
        return surface
    
    # Get recent interactions
    recent_interactions = population.interactions[-20:]
    
    # Count interactions by type
    interaction_counts = {}
    for interaction in recent_interactions:
        interaction_type = interaction["type"]
        if interaction_type not in interaction_counts:
            interaction_counts[interaction_type] = 0
        interaction_counts[interaction_type] += 1
    
    # Sort interaction types by count (descending)
    sorted_types = sorted(interaction_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Define bar properties
    bar_count = len(sorted_types)
    bar_width = width / (bar_count + 1) * 0.8
    bar_colors = [(50, 150, 255), (50, 200, 50), (255, 150, 50), (255, 50, 50)]  # Blue, Green, Orange, Red
    
    # Find maximum value for scaling
    max_value = max(interaction_counts.values()) if interaction_counts else 1
    
    # Draw grid lines
    for i in range(0, height, 20):
        pygame.draw.line(surface, (70, 70, 70), (0, i), (width, i), 1)
    
    # Draw the bars
    font = pygame.font.SysFont("Arial", 12)
    for i, (interaction_type, count) in enumerate(sorted_types):
        # Calculate bar position and height
        x = width * (i + 1) / (bar_count + 1)
        value_height = (count / max_value) * (height - 40)
        bar_rect = pygame.Rect(x - bar_width/2, height - 30 - value_height, bar_width, value_height)
        
        # Draw the bar
        color_index = i % len(bar_colors)
        pygame.draw.rect(surface, bar_colors[color_index], bar_rect)
        pygame.draw.rect(surface, (150, 150, 150), bar_rect, 1)  # Border
        
        # Draw the label
        label = font.render(interaction_type.capitalize(), True, (200, 200, 200))
        label_width = label.get_width()
        
        # If label is too wide, draw at an angle or just center it
        if label_width > bar_width + 20:
            # Simplified approach: just center it
            surface.blit(label, (x - label_width/2, height - 20))
        else:
            surface.blit(label, (x - label_width/2, height - 20))
        
        # Draw the value
        value_text = font.render(f"{count}", True, (200, 200, 200))
        surface.blit(value_text, (x - value_text.get_width()/2, height - 30 - value_height - 15))
    
    # Draw title
    title_font = pygame.font.SysFont("Arial", 14, bold=True)
    title = title_font.render("Recent Interactions by Type", True, (220, 220, 220))
    surface.blit(title, (width//2 - title.get_width()//2, 10))
    
    # Add border
    pygame.draw.rect(surface, (100, 100, 100), (0, 0, width, height), 1)
    
    return surface
