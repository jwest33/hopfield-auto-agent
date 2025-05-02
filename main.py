import pygame
import sys
import os
import pickle
import random
import time
import numpy as np
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import argparse
import threading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Import visualization utilities
from visualization_utils import (
    create_graph_surface, 
    create_population_stats_surface, 
    draw_minimap, 
    create_interaction_graph
)

# Import our modules
from module_new_world import World, TerrainCell
from module_agent import Agent, AgentPopulation
from module_hopfield import Hopfield

# Try to import PyTorch-related modules - handle gracefully if not available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    HAS_TORCH = True
    from module_coms import NeuralCommunicationSystem
    logging.info("PyTorch found - communication features enabled")
except ImportError:
    HAS_TORCH = False
    logging.warning("PyTorch not found - communication features disabled")
    
    # Create a minimal communication system substitute
    class DummyCommSystem:
        def __init__(self):
            self.active_signals = []
        def clean_old_signals(self):
            pass
        def initialize_agent(self, agent_id):
            pass

# ───────────────────── Constants ─────────────────────
# Simulation parameters
GRID = 40
FPS = 30
CELL_SIZE = 18
PANEL_WIDTH = 500  # Increased panel width
WINDOW_WIDTH = GRID * CELL_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = 800  # Increased window height
CLOCK_SPEED = 10  # Simulation ticks per second at normal speed

# File paths
STATE_FILE = "agent_state.npz"
WORLD_FILE = "world.pkl"
SAVE_DIR = "saves"

# Material colors
MATERIAL_COLORS = {
    "dirt": (210, 180, 140),
    "stone": (169, 169, 169),
    "rock": (128, 128, 128),
    "water": (30, 144, 255),
    "wood": (34, 139, 34),
    "food": (220, 20, 60),
    "home": (255, 215, 0),
    "empty": (230, 230, 230),
}

# UI colors
COLOR_BG = (30, 30, 30)
COLOR_PANEL = (50, 50, 50)
COLOR_TEXT = (240, 240, 240)
COLOR_TEXT_HEADER = (255, 255, 255)
COLOR_BUTTON = (80, 80, 80)
COLOR_BUTTON_HOVER = (100, 100, 100)
COLOR_BUTTON_ACTIVE = (120, 120, 120)

# Button states
BUTTON_NORMAL = 0
BUTTON_HOVER = 1
BUTTON_PRESSED = 2

# ───────────────────── Helper Functions ─────────────────────
def load_or_create_world(grid_size=GRID):
    """Load an existing world or create a new one"""
    if os.path.exists(WORLD_FILE):
        try:
            with open(WORLD_FILE, "rb") as f:
                world = pickle.load(f)
                logging.info(f"Loaded world from {WORLD_FILE}")
                return world
        except Exception as e:
            logging.error(f"Error loading world: {e}")
    
    # If no file exists or loading failed, create a new world
    world = World(grid_size)
    logging.info(f"Created new world with grid size {grid_size}")
    return world

def is_food_cell(cell: TerrainCell) -> bool:
    """Check if a cell contains food"""
    return (cell.material == "food" or "food" in cell.tags)

def save_simulation(world, population, filename=None):
    """Save the current simulation state"""
    # Create save directory if it doesn't exist
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sim_{timestamp}.pkl"
    
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Save the simulation state
    try:
        with open(filepath, "wb") as f:
            pickle.dump({"world": world, "population": population}, f)
        logging.info(f"Simulation saved to {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving simulation: {e}")
        return False

def load_simulation(filepath):
    """Load a saved simulation state"""
    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            world = data["world"]
            population = data["population"]
        logging.info(f"Simulation loaded from {filepath}")
        return world, population
    except Exception as e:
        logging.error(f"Error loading simulation: {e}")
        return None, None

# ───────────────────── UI Components ─────────────────────
class Button:
    """Interactive button with hover and click states"""
    def __init__(self, rect, text, callback=None, color=COLOR_BUTTON, 
                 hover_color=COLOR_BUTTON_HOVER, active_color=COLOR_BUTTON_ACTIVE):
        self.rect = pygame.Rect(rect)
        self.text = text
        self.callback = callback
        self.color = color
        self.hover_color = hover_color
        self.active_color = active_color
        self.state = BUTTON_NORMAL
    
    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        self.state = BUTTON_NORMAL
        
        if self.rect.collidepoint(mouse_pos):
            self.state = BUTTON_HOVER
            
            for event in events:
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.state = BUTTON_PRESSED
                elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                    if self.callback and self.rect.collidepoint(mouse_pos):
                        self.callback()
    
    def draw(self, screen, font):
        # Draw button background based on state
        if self.state == BUTTON_NORMAL:
            color = self.color
        elif self.state == BUTTON_HOVER:
            color = self.hover_color
        else:
            color = self.active_color
            
        pygame.draw.rect(screen, color, self.rect, border_radius=5)
        pygame.draw.rect(screen, (200, 200, 200), self.rect, 2, border_radius=5)
        
        # Draw button text
        text_surf = font.render(self.text, True, COLOR_TEXT)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

class Slider:
    """Interactive slider for adjusting simulation parameters"""
    def __init__(self, rect, min_val, max_val, value, label, callback=None):
        self.rect = pygame.Rect(rect)
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.label = label
        self.callback = callback
        self.dragging = False
        self.handle_width = 12
        
    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                handle_rect = self.get_handle_rect()
                if handle_rect.collidepoint(mouse_pos):
                    self.dragging = True
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
        
        if self.dragging:
            # Calculate value based on mouse x position
            x_pos = max(self.rect.left, min(mouse_pos[0], self.rect.right))
            ratio = (x_pos - self.rect.left) / self.rect.width
            self.value = self.min_val + ratio * (self.max_val - self.min_val)
            
            # Call the callback if provided
            if self.callback:
                self.callback(self.value)
    
    def get_handle_rect(self):
        """Get the Rectangle for the slider handle"""
        ratio = (self.value - self.min_val) / (self.max_val - self.min_val)
        handle_x = self.rect.left + ratio * self.rect.width - self.handle_width // 2
        return pygame.Rect(handle_x, self.rect.top - 6, self.handle_width, self.rect.height + 12)
    
    def draw(self, screen, font):
        # Draw slider track
        pygame.draw.rect(screen, (80, 80, 80), self.rect, border_radius=3)
        
        # Draw slider handle
        handle_rect = self.get_handle_rect()
        pygame.draw.rect(screen, (160, 160, 160), handle_rect, border_radius=4)
        pygame.draw.rect(screen, (200, 200, 200), handle_rect, 2, border_radius=4)
        
        # Draw label and value - moved up to prevent overlap
        label_surf = font.render(f"{self.label}: {self.value:.2f}", True, COLOR_TEXT)
        label_rect = label_surf.get_rect(midleft=(self.rect.left, self.rect.top - 15))
        screen.blit(label_surf, label_rect)

class ScrollableArea:
    """Area that can be scrolled vertically"""
    def __init__(self, rect, content_height):
        self.rect = pygame.Rect(rect)
        self.content_height = content_height
        self.scroll_y = 0
        self.scroll_speed = 20
        self.visible_rect = pygame.Rect(rect.left, rect.top, rect.width, rect.height)
        self.dragging = False
        self.last_mouse_y = 0
        
    def update(self, events):
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    self.scroll_y = max(0, self.scroll_y - self.scroll_speed)
                elif event.button == 5:  # Mouse wheel down
                    max_scroll = max(0, self.content_height - self.rect.height)
                    self.scroll_y = min(max_scroll, self.scroll_y + self.scroll_speed)
                elif event.button == 1 and self.rect.collidepoint(event.pos):  # Left click
                    self.dragging = True
                    self.last_mouse_y = event.pos[1]
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.dragging = False
            elif event.type == pygame.MOUSEMOTION and self.dragging:
                dy = event.pos[1] - self.last_mouse_y
                self.scroll_y = max(0, min(self.content_height - self.rect.height, self.scroll_y - dy))
                self.last_mouse_y = event.pos[1]
                
        # Update the visible rectangle
        self.visible_rect = pygame.Rect(self.rect.left, self.rect.top - self.scroll_y, 
                                        self.rect.width, self.rect.height)
    
    def get_content_rect(self, height):
        """Get a rectangle for content relative to the scrollable area"""
        return pygame.Rect(self.rect.left, self.rect.top + height - self.scroll_y,
                         self.rect.width, height)
    
    def draw_scrollbar(self, screen):
        if self.content_height <= self.rect.height:
            return  # No scrollbar needed
            
        # Calculate scrollbar dimensions
        scrollbar_height = max(30, self.rect.height * (self.rect.height / self.content_height))
        scrollbar_pos = self.rect.right - 10
        scrollbar_top = self.rect.top + (self.scroll_y / self.content_height) * self.rect.height
        
        # Draw scrollbar track
        pygame.draw.rect(screen, (60, 60, 60), 
                      (scrollbar_pos - 2, self.rect.top, 
                       14, self.rect.height), 
                      border_radius=7)
        
        # Draw scrollbar handle
        pygame.draw.rect(screen, (140, 140, 140), 
                      (scrollbar_pos, scrollbar_top, 
                       10, scrollbar_height), 
                      border_radius=5)

class ProgressBar:
    """Progress bar for displaying agent stats"""
    def __init__(self, rect, value, max_value, label=None, color=(0, 200, 0)):
        self.rect = pygame.Rect(rect)
        self.value = value
        self.max_value = max_value
        self.label = label
        self.color = color
    
    def update(self, value):
        self.value = value
    
    def draw(self, screen, font):
        # Draw background
        pygame.draw.rect(screen, (60, 60, 60), self.rect, border_radius=3)
        
        # Draw filled portion
        fill_width = int(self.rect.width * (self.value / self.max_value))
        fill_rect = pygame.Rect(self.rect.left, self.rect.top, fill_width, self.rect.height)
        pygame.draw.rect(screen, self.color, fill_rect, border_radius=3)
        
        # Draw border
        pygame.draw.rect(screen, (120, 120, 120), self.rect, 1, border_radius=3)
        
        # Draw label and value if provided
        if self.label:
            text = f"{self.label}: {self.value:.1f}/{self.max_value}"
            text_surf = font.render(text, True, COLOR_TEXT)
            text_rect = text_surf.get_rect(midleft=(self.rect.left, self.rect.top - 10))
            screen.blit(text_surf, text_rect)

class TabControl:
    """Tabbed interface for organizing content"""
    def __init__(self, rect, tabs):
        self.rect = pygame.Rect(rect)
        self.tabs = tabs  # List of tab names
        self.active_tab = 0
        self.tab_width = self.rect.width // len(tabs)
        self.tab_height = 30
        
    def update(self, events):
        mouse_pos = pygame.mouse.get_pos()
        
        # Check for tab clicks
        for i, tab in enumerate(self.tabs):
            tab_rect = self.get_tab_rect(i)
            if tab_rect.collidepoint(mouse_pos):
                for event in events:
                    if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                        self.active_tab = i
                        break
    
    def get_tab_rect(self, tab_index):
        """Get rectangle for a specific tab"""
        return pygame.Rect(
            self.rect.left + tab_index * self.tab_width,
            self.rect.top,
            self.tab_width,
            self.tab_height
        )
    
    def get_content_rect(self):
        """Get rectangle for the tab content area"""
        return pygame.Rect(
            self.rect.left,
            self.rect.top + self.tab_height,
            self.rect.width,
            self.rect.height - self.tab_height
        )
    
    def draw(self, screen, font):
        # Draw tab backgrounds
        for i, tab in enumerate(self.tabs):
            tab_rect = self.get_tab_rect(i)
            
            # Different styling for active tab
            if i == self.active_tab:
                # Active tab
                pygame.draw.rect(screen, (80, 80, 80), tab_rect)
                # Bottom line in active tab color to create connected look
                pygame.draw.line(
                    screen, 
                    (80, 80, 80), 
                    (tab_rect.left, tab_rect.bottom - 1),
                    (tab_rect.right, tab_rect.bottom - 1),
                    2
                )
            else:
                # Inactive tab
                pygame.draw.rect(screen, (60, 60, 60), tab_rect)
                
            # Border
            pygame.draw.rect(screen, (120, 120, 120), tab_rect, 1)
            
            # Tab text
            tab_text = font.render(tab, True, COLOR_TEXT)
            text_rect = tab_text.get_rect(center=tab_rect.center)
            screen.blit(tab_text, text_rect)
        
        # Draw content area
        content_rect = self.get_content_rect()
        pygame.draw.rect(screen, (80, 80, 80), content_rect)
        pygame.draw.rect(screen, (120, 120, 120), content_rect, 1)
        
        return content_rect

# ───────────────────── Main Simulation Class ─────────────────────
class Simulation:
    """Main simulation class handling the game loop and UI"""
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Multi-Agent Emergence Simulation")
        
        # Set up the screen
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.world = load_or_create_world(GRID)
        
        # Initialize population with appropriate communication system
        self.population = AgentPopulation(self.world, initial_pop=5)
        if not HAS_TORCH:
            self.population.comm_system = DummyCommSystem()
        
        # Set up simulation state
        self.running = True
        self.paused = True
        self.sim_speed = 1.0
        self.step_manually = False
        self.tick_count = 0
        self.last_tick_time = time.time()
        self.selected_agent_id = list(self.population.agents.keys())[0] if self.population.agents else None
        
        # Initialize fonts
        self.font_small = pygame.font.SysFont("Arial", 12)
        self.font_normal = pygame.font.SysFont("Arial", 14)
        self.font_large = pygame.font.SysFont("Arial", 18)
        self.font_header = pygame.font.SysFont("Arial", 22, bold=True)
        
        # Set up UI components
        self.setup_ui()
        
        # Set up simulation clock
        self.clock = pygame.time.Clock()
        
        # Show minimap by default
        self.show_minimap = True
        
        logging.info("Simulation initialized")
        
    def setup_ui(self):
        """Set up all UI components"""
        margin = 15
        panel_left = GRID * CELL_SIZE + margin
        panel_width = PANEL_WIDTH - 2 * margin
        button_height = 30
        
        # Calculate the grid display area size
        grid_area_size = min(WINDOW_HEIGHT, GRID * CELL_SIZE)
        
        # Main control buttons - positions will be updated in render_panel
        self.btn_play = Button(
            (panel_left, margin, panel_width//2 - 5, button_height),
            "▶ Play" if self.paused else "⏸ Pause",
            self.toggle_pause
        )
        
        self.btn_step = Button(
            (panel_left + panel_width//2 + 5, margin, panel_width//2 - 5, button_height),
            "Step",
            self.step_once
        )
        
        # Speed slider - moved down by adding more vertical space
        slider_top = margin + button_height + 25  # Increased spacing from 15 to 25
        self.slider_speed = Slider(
            (panel_left, slider_top, panel_width, 10),
            0.1, 5.0, self.sim_speed, "Simulation Speed",
            self.set_speed
        )
        
        # Agent management buttons - adjusted position
        agent_btn_top = slider_top + 50  # Increased from 40 to 50 for more space
        self.btn_add_agent = Button(
            (panel_left, agent_btn_top, panel_width//2 - 5, button_height),
            "Add Agent",
            self.add_agent
        )
        
        self.btn_remove_agent = Button(
            (panel_left + panel_width//2 + 5, agent_btn_top, panel_width//2 - 5, button_height),
            "Remove Agent",
            self.remove_selected_agent
        )
        
        # Save/Load buttons - adjusted position
        save_btn_top = agent_btn_top + button_height + 15  # Increased from 10 to 15
        self.btn_save = Button(
            (panel_left, save_btn_top, panel_width//2 - 5, button_height),
            "Save Simulation",
            self.save_simulation
        )
        
        self.btn_load = Button(
            (panel_left + panel_width//2 + 5, save_btn_top, panel_width//2 - 5, button_height),
            "Load Simulation",
            self.load_simulation
        )
        
        # Reset button - adjusted position
        reset_btn_top = save_btn_top + button_height + 15  # Increased from 10 to 15
        self.btn_reset = Button(
            (panel_left, reset_btn_top, panel_width, button_height),
            "Reset Simulation",
            self.reset_simulation
        )
        
        # Toggle minimap button - adjusted position
        minimap_btn_top = reset_btn_top + button_height + 15  # Increased from 10 to 15
        self.btn_toggle_minimap = Button(
            (panel_left, minimap_btn_top, panel_width, button_height),
            "Toggle Minimap",
            self.toggle_minimap
        )
        
        # Agent dropdown - select box for agents - adjusted position
        agent_selection_top = minimap_btn_top + button_height + 30  # Increased from 25 to 30
        self.agent_selection_rect = pygame.Rect(panel_left, agent_selection_top, panel_width, button_height)
        self.agent_dropdown_open = False
        self.agent_dropdown_rect = pygame.Rect(panel_left, agent_selection_top + button_height, panel_width, 0)
        
        # World info display
        world_info_top = agent_selection_top + button_height + 25
        
        # Tab control for statistics views - positioned below world info
        tab_top = world_info_top + 100  # Leave room for world info
        self.tabs = TabControl(
            (panel_left, tab_top, panel_width, WINDOW_HEIGHT - tab_top - margin),
            ["Agent", "Population", "Interactions"]
        )
        
        # Scrollable area for agent stats
        content_rect = self.tabs.get_content_rect()
        self.stats_area = ScrollableArea(
            content_rect,
            700  # Initial content height
        )
        
        # Progress bars for agent stats - will be updated in render_agent_stats
        self.agent_progress_bars = {}
        
        # Cached visualization surfaces
        self.visualization_cache = {
            "agent_graph": None,
            "population_stats": None,
            "interaction_graph": None,
            "minimap": None,
            "cache_tick": -1  # Tick when cache was last updated
        }
    
    def toggle_pause(self):
        """Toggle the simulation pause state"""
        self.paused = not self.paused
        self.btn_play.text = "▶ Play" if self.paused else "⏸ Pause"
        logging.info(f"Simulation {'paused' if self.paused else 'resumed'}")
    
    def set_speed(self, speed):
        """Set the simulation speed"""
        self.sim_speed = speed
    
    def step_once(self):
        """Step the simulation a single tick"""
        self.step_manually = True
    
    def add_agent(self):
        """Add a new agent to the simulation"""
        agent = self.population.add_agent()
        if agent:
            self.selected_agent_id = agent.id
            logging.info(f"Added agent {agent.id}")
    
    def remove_selected_agent(self):
        """Remove the currently selected agent"""
        if len(self.population.agents) > 1 and self.selected_agent_id:
            self.population.remove_agent(self.selected_agent_id)
            self.selected_agent_id = list(self.population.agents.keys())[0]
            logging.info(f"Removed agent {self.selected_agent_id}")
    
    def toggle_minimap(self):
        """Toggle the minimap display"""
        self.show_minimap = not self.show_minimap
    
    def save_simulation(self):
        """Save the current simulation state"""
        save_simulation(self.world, self.population)
    
    def load_simulation(self):
        """Load a saved simulation state"""
        if not os.path.exists(SAVE_DIR):
            logging.warning("No saves directory found")
            return
            
        save_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pkl")]
        if not save_files:
            logging.warning("No save files found")
            return
            
        # For simplicity, just load the most recent save
        save_files.sort(reverse=True)
        most_recent = os.path.join(SAVE_DIR, save_files[0])
        
        world, population = load_simulation(most_recent)
        if world and population:
            self.world = world
            self.population = population
            self.selected_agent_id = list(self.population.agents.keys())[0] if self.population.agents else None
            
            # Clear visualization cache
            self.visualization_cache = {
                "agent_graph": None,
                "population_stats": None,
                "interaction_graph": None,
                "minimap": None,
                "cache_tick": -1
            }
    
    def reset_simulation(self):
        """Reset the simulation to a fresh state"""
        self.world = load_or_create_world(GRID)
        self.population = AgentPopulation(self.world, initial_pop=5)
        if not HAS_TORCH:
            self.population.comm_system = DummyCommSystem()
        self.selected_agent_id = list(self.population.agents.keys())[0] if self.population.agents else None
        self.tick_count = 0
        self.paused = True
        self.btn_play.text = "▶ Play"
        
        # Clear visualization cache
        self.visualization_cache = {
            "agent_graph": None,
            "population_stats": None,
            "interaction_graph": None,
            "minimap": None,
            "cache_tick": -1
        }
        
        logging.info("Simulation reset")
    
    def process_events(self):
        """Process pygame events"""
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_SPACE:
                    self.toggle_pause()
                elif event.key == pygame.K_RIGHT:
                    self.step_once()
                elif event.key == pygame.K_m:
                    self.toggle_minimap()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Handle agent selection from grid
                    if event.pos[0] < GRID * CELL_SIZE:
                        cell_x = event.pos[1] // CELL_SIZE
                        cell_y = event.pos[0] // CELL_SIZE
                        
                        for agent_id, agent in self.population.agents.items():
                            ax, ay = agent.pos
                            if ax == cell_x and ay == cell_y:
                                self.selected_agent_id = agent_id
                                # Clear agent graph cache on agent change
                                self.visualization_cache["agent_graph"] = None
                                break
                    
                    # Handle agent dropdown
                    if self.agent_selection_rect.collidepoint(event.pos):
                        self.agent_dropdown_open = not self.agent_dropdown_open
                    elif self.agent_dropdown_open:
                        # Check if clicked on an agent in the dropdown
                        dropdown_y = self.agent_selection_rect.bottom
                        for i, agent_id in enumerate(self.population.agents.keys()):
                            agent_rect = pygame.Rect(
                                self.agent_selection_rect.left, 
                                dropdown_y + i * 25, 
                                self.agent_selection_rect.width, 
                                25
                            )
                            if agent_rect.collidepoint(event.pos):
                                self.selected_agent_id = agent_id
                                self.agent_dropdown_open = False
                                # Clear agent graph cache on agent change
                                self.visualization_cache["agent_graph"] = None
                                break
                        else:
                            # Close dropdown if clicked outside
                            self.agent_dropdown_open = False
        
        # Update UI components
        self.btn_play.update(events)
        self.btn_step.update(events)
        self.slider_speed.update(events)
        self.btn_add_agent.update(events)
        self.btn_remove_agent.update(events)
        self.btn_save.update(events)
        self.btn_load.update(events)
        self.btn_reset.update(events)
        self.btn_toggle_minimap.update(events)
        self.tabs.update(events)
        self.stats_area.update(events)
    
    def update_caches(self):
        """Update visualization caches if needed"""
        # Only update caches every 30 ticks (about once per second) to save performance
        if self.tick_count % 30 != 0 and self.visualization_cache["cache_tick"] > 0:
            return
            
        # Update agent graph
        if self.selected_agent_id in self.population.agents:
            agent = self.population.agents[self.selected_agent_id]
            if len(agent.history["energy"]) > 10:
                self.visualization_cache["agent_graph"] = create_graph_surface(
                    agent, 
                    width=self.stats_area.rect.width, 
                    height=150
                )
        
        # Update population stats
        if len(self.population.agents) > 1:
            self.visualization_cache["population_stats"] = create_population_stats_surface(
                self.population,
                width=self.stats_area.rect.width,
                height=150
            )
        
        # Update interaction graph
        if len(self.population.interactions) > 3:
            self.visualization_cache["interaction_graph"] = create_interaction_graph(
                self.population,
                width=self.stats_area.rect.width,
                height=150
            )
        
        # Update minimap cache
        self.visualization_cache["minimap"] = draw_minimap(
            pygame.Surface((120, 120)), 
            self.world, 
            size=120
        )
        
        # Update cache tick
        self.visualization_cache["cache_tick"] = self.tick_count
    
    def update(self):
        """Update the simulation state"""
        current_time = time.time()
        step_time = 1.0 / (CLOCK_SPEED * self.sim_speed)
        
        # Check if we should step the simulation
        if (not self.paused or self.step_manually) and current_time - self.last_tick_time >= step_time:
            # Step the world
            self.world.step()
            
            # Step all agents
            self.population.step_all()
            
            self.last_tick_time = current_time
            self.tick_count += 1
            self.step_manually = False
        
        # Update visualization caches
        self.update_caches()
        
        # Update agent dropdown height
        if self.agent_dropdown_open:
            self.agent_dropdown_rect.height = len(self.population.agents) * 25
        else:
            self.agent_dropdown_rect.height = 0
    
    def render(self):
        """Render the simulation to the screen"""
        # Clear the screen
        self.screen.fill(COLOR_BG)
        
        # Draw the world grid
        self.render_world()
        
        # Draw the side panel
        self.render_panel()
        
        # Draw minimap if enabled
        if self.show_minimap and self.visualization_cache["minimap"]:
            self.screen.blit(self.visualization_cache["minimap"], (20, 20))
        
        # Flip the display
        pygame.display.flip()
    
    def render_world(self):
        """Render the world grid"""
        # Draw grid background
        grid_rect = pygame.Rect(0, 0, GRID * CELL_SIZE, GRID * CELL_SIZE)
        pygame.draw.rect(self.screen, (230, 230, 230), grid_rect)
        
        # Draw cell types
        for i in range(GRID):
            for j in range(GRID):
                cell = self.world.cell((i, j))
                color = MATERIAL_COLORS.get(cell.material, (200, 200, 200))
                
                # Special case for food cells
                if is_food_cell(cell):
                    rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    # Draw the base square with food color
                    pygame.draw.rect(self.screen, color, rect)
                    pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)
                    # Add a darker red circle in the center
                    pygame.draw.circle(
                        self.screen, 
                        (180, 0, 0),
                        (j * CELL_SIZE + CELL_SIZE//2, i * CELL_SIZE + CELL_SIZE//2), 
                        CELL_SIZE//3
                    )
                    continue  # Skip normal cell drawing since we drew a custom design
                # Special case for home cells
                elif cell.material == "home" or "home" in cell.tags:
                    color = MATERIAL_COLORS["home"]
                
                rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (180, 180, 180), rect, 1)
        
        # Draw home marker
        hx, hy = self.world.home
        home_rect = pygame.Rect(hy * CELL_SIZE, hx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        home_icon = self.font_large.render("H", True, (0, 0, 0))
        self.screen.blit(home_icon, home_icon.get_rect(center=home_rect.center))
        
        # Draw agents
        agent_icons = ["1", "2", "3", "4", "5", "6"]
        for i, (agent_id, agent) in enumerate(self.population.agents.items()):
            ax, ay = agent.pos
            agent_rect = pygame.Rect(ay * CELL_SIZE, ax * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            
            # Highlight selected agent
            if agent_id == self.selected_agent_id:
                pygame.draw.rect(self.screen, (100, 150, 255), agent_rect, 3)
            
            # Draw agent icon
            icon_idx = i % len(agent_icons)
            # Use simple circle with color based on agent index
            agent_colors = [(50, 150, 255), (255, 150, 50), (100, 200, 100), 
                          (200, 100, 200), (100, 200, 200), (200, 200, 100)]
            agent_color = agent_colors[i % len(agent_colors)]
            
            # Draw agent as a colored circle with an ID number
            pygame.draw.circle(
                self.screen, 
                agent_color,
                (agent_rect.centerx, agent_rect.centery), 
                CELL_SIZE//2 - 2
            )
            # Draw agent ID number
            id_num = agent_id.split('_')[1]
            id_surf = self.font_small.render(id_num, True, (0, 0, 0))
            id_rect = id_surf.get_rect(center=agent_rect.center)
            self.screen.blit(id_surf, id_rect)
            
            # Draw carrying indicator if agent is carrying something
            if agent.carrying:
                # Draw a smaller red circle at top-right
                carry_pos = (agent_rect.right - CELL_SIZE//4, agent_rect.top + CELL_SIZE//4)
                pygame.draw.circle(
                    self.screen, 
                    (220, 20, 60),  # Red
                    carry_pos, 
                    CELL_SIZE//4
                )
        
        # Draw communication signals
        if HAS_TORCH:
            for signal in self.population.comm_system.active_signals:
                sx, sy = signal["position"]
                signal_rect = pygame.Rect(sy * CELL_SIZE, sx * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                # Draw signal indicator
                s = pygame.Surface((CELL_SIZE, CELL_SIZE), pygame.SRCALPHA)
                pygame.draw.circle(s, (255, 255, 100, 128), (CELL_SIZE//2, CELL_SIZE//2), CELL_SIZE//2)
                self.screen.blit(s, signal_rect)
                
                signal_icon = self.font_normal.render("💬", True, (0, 0, 0))
                self.screen.blit(signal_icon, signal_icon.get_rect(center=signal_rect.center))

    def render_panel(self):
        """Render the control panel"""
        # Draw panel background
        panel_rect = pygame.Rect(GRID * CELL_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel_rect)
        
        # Draw title
        title_surf = self.font_header.render("Multi-Agent Simulation", True, COLOR_TEXT_HEADER)
        # Position title 10px from the top of the panel
        title_rect = title_surf.get_rect(midtop=(GRID * CELL_SIZE + PANEL_WIDTH // 2, 10))
        self.screen.blit(title_surf, title_rect)
        
        # Adjust button positions to be below the title
        button_top = title_rect.bottom + 15  # 15px spacing after title
        self.btn_play.rect.top = button_top
        self.btn_step.rect.top = button_top
        
        # Update slider position to ensure it doesn't overlap with buttons
        self.slider_speed.rect.top = button_top + self.btn_play.rect.height + 25
        
        # Draw UI controls
        self.btn_play.draw(self.screen, self.font_normal)
        self.btn_step.draw(self.screen, self.font_normal)
        self.slider_speed.draw(self.screen, self.font_normal)
        self.btn_add_agent.draw(self.screen, self.font_normal)
        self.btn_remove_agent.draw(self.screen, self.font_normal)
        self.btn_save.draw(self.screen, self.font_normal)
        self.btn_load.draw(self.screen, self.font_normal)
        self.btn_reset.draw(self.screen, self.font_normal)
        self.btn_toggle_minimap.draw(self.screen, self.font_normal)
        
        # Draw agent selector
        self.render_agent_selector()
        
        # Draw world info
        world_info_y = self.agent_selection_rect.bottom + (25 if not self.agent_dropdown_open else self.agent_dropdown_rect.height + 25)
        world_info = [
            f"Tick: {self.tick_count}",
            f"Time: {'Day' if self.world.is_day else 'Night'}",
            f"Weather: {self.world.weather.capitalize()}",
            f"Temperature: {self.world.ambient_temperature:.1f}°C",
            f"Active Agents: {len(self.population.agents)}"
        ]
        
        # Create a background for world info
        world_info_height = len(world_info) * 20 + 10
        world_info_rect = pygame.Rect(
            panel_rect.left + 10, 
            world_info_y, 
            panel_rect.width - 20, 
            world_info_height
        )
        pygame.draw.rect(self.screen, (60, 60, 60), world_info_rect, border_radius=5)
        pygame.draw.rect(self.screen, (100, 100, 100), world_info_rect, 1, border_radius=5)
        
        # Draw world info text
        for i, info in enumerate(world_info):
            info_surf = self.font_normal.render(info, True, COLOR_TEXT)
            info_rect = info_surf.get_rect(topleft=(panel_rect.left + 20, world_info_y + 10 + i * 20))
            self.screen.blit(info_surf, info_rect)
        
        # Adjust tab position based on world info position
        self.tabs.rect.top = world_info_y + world_info_height + 15
        self.tabs.rect.height = WINDOW_HEIGHT - self.tabs.rect.top - 10
        
        # Draw tabs
        content_rect = self.tabs.draw(self.screen, self.font_normal)
        
        # Update stats area rectangle to match tab content
        self.stats_area.rect = content_rect
        
        # Draw content based on active tab
        if self.tabs.active_tab == 0:  # Agent tab
            if self.selected_agent_id in self.population.agents:
                self.render_agent_stats()
            else:
                # No agent selected
                info = self.font_normal.render("No agent selected", True, COLOR_TEXT)
                self.screen.blit(info, info.get_rect(center=content_rect.center))
        elif self.tabs.active_tab == 1:  # Population tab
            self.render_population_stats()
        else:  # Interactions tab
            self.render_interaction_stats()
        
        # Draw scrollbar if needed
        self.stats_area.draw_scrollbar(self.screen)
    
    def render_agent_selector(self):
        """Render the agent selection dropdown"""
        # Draw the selection box
        selected_agent_num = self.selected_agent_id.split('_')[1] if self.selected_agent_id else "None"
        text = f"Selected Agent: {selected_agent_num}"
        
        pygame.draw.rect(self.screen, (80, 80, 80), self.agent_selection_rect, border_radius=5)
        pygame.draw.rect(self.screen, (150, 150, 150), self.agent_selection_rect, 1, border_radius=5)
        
        # Draw current selection
        text_surf = self.font_normal.render(text, True, COLOR_TEXT)
        text_rect = text_surf.get_rect(midleft=(self.agent_selection_rect.left + 10, self.agent_selection_rect.centery))
        self.screen.blit(text_surf, text_rect)
        
        # Draw dropdown arrow
        arrow = "▼" if not self.agent_dropdown_open else "▲"
        arrow_surf = self.font_normal.render(arrow, True, COLOR_TEXT)
        arrow_rect = arrow_surf.get_rect(midright=(self.agent_selection_rect.right - 10, self.agent_selection_rect.centery))
        self.screen.blit(arrow_surf, arrow_rect)
        
        # Draw dropdown if open
        if self.agent_dropdown_open:
            dropdown_rect = pygame.Rect(
                self.agent_selection_rect.left,
                self.agent_selection_rect.bottom,
                self.agent_selection_rect.width,
                len(self.population.agents) * 25
            )
            pygame.draw.rect(self.screen, (100, 100, 100), dropdown_rect)
            pygame.draw.rect(self.screen, (150, 150, 150), dropdown_rect, 1)
            
            # Draw agent options
            for i, agent_id in enumerate(self.population.agents.keys()):
                agent_num = agent_id.split('_')[1]
                option_rect = pygame.Rect(
                    dropdown_rect.left,
                    dropdown_rect.top + i * 25,
                    dropdown_rect.width,
                    25
                )
                
                # Highlight selected agent
                if agent_id == self.selected_agent_id:
                    pygame.draw.rect(self.screen, (80, 100, 120), option_rect)
                
                # Draw agent text
                agent_text = f"Agent {agent_num}"
                agent_surf = self.font_normal.render(agent_text, True, COLOR_TEXT)
                agent_rect = agent_surf.get_rect(midleft=(option_rect.left + 10, option_rect.centery))
                self.screen.blit(agent_surf, agent_rect)
    
    def render_agent_stats(self):
        """Render the selected agent's statistics"""
        if self.selected_agent_id not in self.population.agents:
            return
            
        agent = self.population.agents[self.selected_agent_id]
        
        # Set up variables for positioning
        content_height = 0
        line_height = 20
        progress_height = 15
        section_spacing = 10
        agent_id_num = self.selected_agent_id.split('_')[1]
        
        # Clear the agent progress bars dictionary
        self.agent_progress_bars = {}
        
        # Agent header with proper positioning
        agent_header = self.font_large.render(f"Agent {agent_id_num}", True, COLOR_TEXT_HEADER)
        header_rect = self.stats_area.get_content_rect(content_height)
        if header_rect.top > 0 and header_rect.top < WINDOW_HEIGHT:
            self.screen.blit(agent_header, header_rect)
        content_height += line_height * 2  # Increased spacing after header
        
        # Agent vitals
        vitals = [
            {"name": "Energy", "value": agent.energy, "max": 100, "color": (100, 200, 100)},
            {"name": "Hunger", "value": agent.hunger, "max": 100, "color": (200, 150, 100)},
            {"name": "Pain", "value": agent.pain, "max": 100, "color": (200, 100, 100)}
        ]
        
        for vital in vitals:
            progress_rect = pygame.Rect(
                self.stats_area.rect.left,
                self.stats_area.rect.top + content_height - self.stats_area.scroll_y,
                self.stats_area.rect.width,
                progress_height
            )
            
            if progress_rect.top > 0 and progress_rect.top < WINDOW_HEIGHT:
                progress_bar = ProgressBar(
                    progress_rect,
                    vital["value"],
                    vital["max"],
                    vital["name"],
                    vital["color"]
                )
                progress_bar.draw(self.screen, self.font_small)
            
            content_height += progress_height + 15
        
        # Food information
        food_info = [
            f"Food Stored: {agent.store}",
            f"Carrying Food: {'Yes' if agent.carrying else 'No'}"
        ]
        
        content_height += section_spacing
        for info in food_info:
            info_rect = self.stats_area.get_content_rect(content_height)
            if info_rect.top > 0 and info_rect.top < WINDOW_HEIGHT:
                info_surf = self.font_normal.render(info, True, COLOR_TEXT)
                self.screen.blit(info_surf, info_rect)
            content_height += line_height
        
        # Environment information
        content_height += section_spacing
        env_header = self.font_normal.render("Environment", True, COLOR_TEXT_HEADER)
        env_header_rect = self.stats_area.get_content_rect(content_height)
        if env_header_rect.top > 0 and env_header_rect.top < WINDOW_HEIGHT:
            self.screen.blit(env_header, env_header_rect)
        content_height += line_height
        
        current_cell = self.world.cell(tuple(agent.pos))
        env_info = [
            f"Position: ({agent.pos[0]}, {agent.pos[1]})",
            f"Terrain: {current_cell.material.title()}",
            f"Temperature: {current_cell.temperature:.1f}°C",
            f"Passable: {'Yes' if current_cell.passable else 'No'}",
            f"Risk: {current_cell.local_risk * 100:.1f}%"
        ]
        
        for info in env_info:
            info_rect = self.stats_area.get_content_rect(content_height)
            if info_rect.top > 0 and info_rect.top < WINDOW_HEIGHT:
                info_surf = self.font_normal.render(info, True, COLOR_TEXT)
                self.screen.blit(info_surf, info_rect)
            content_height += line_height
        
        # Action information
        content_height += section_spacing
        action_header = self.font_normal.render("Actions", True, COLOR_TEXT_HEADER)
        action_header_rect = self.stats_area.get_content_rect(content_height)
        if action_header_rect.top > 0 and action_header_rect.top < WINDOW_HEIGHT:
            self.screen.blit(action_header, action_header_rect)
        content_height += line_height
        
        action_info = [
            f"Last Action: {agent.last_action}",
            f"Last Reward: {agent.last_reward:.1f}",
            f"Tick Count: {agent.tick_count}"
        ]
        
        for info in action_info:
            info_rect = self.stats_area.get_content_rect(content_height)
            if info_rect.top > 0 and info_rect.top < WINDOW_HEIGHT:
                info_surf = self.font_normal.render(info, True, COLOR_TEXT)
                self.screen.blit(info_surf, info_rect)
            content_height += line_height
        
        # Communication information (if PyTorch available)
        if HAS_TORCH:
            content_height += section_spacing
            comm_header = self.font_normal.render("Communication", True, COLOR_TEXT_HEADER)
            comm_header_rect = self.stats_area.get_content_rect(content_height)
            if comm_header_rect.top > 0 and comm_header_rect.top < WINDOW_HEIGHT:
                self.screen.blit(comm_header, comm_header_rect)
            content_height += line_height
            
            comm_info = [
                f"Signals Sent: {agent.last_signal_time if agent.last_signal is not None else 0}",
                f"Signal Cooldown: {agent.signal_cooldown}"
            ]
            
            for info in comm_info:
                info_rect = self.stats_area.get_content_rect(content_height)
                if info_rect.top > 0 and info_rect.top < WINDOW_HEIGHT:
                    info_surf = self.font_normal.render(info, True, COLOR_TEXT)
                    self.screen.blit(info_surf, info_rect)
                content_height += line_height
        
        # Agent history graph
        content_height += section_spacing * 2
        if self.visualization_cache["agent_graph"] is not None:
            graph_rect = self.visualization_cache["agent_graph"].get_rect(
                topleft=(self.stats_area.rect.left, 
                        self.stats_area.rect.top + content_height - self.stats_area.scroll_y)
            )
            
            if graph_rect.top + 20 > 0 and graph_rect.top < WINDOW_HEIGHT:
                self.screen.blit(self.visualization_cache["agent_graph"], graph_rect)
            
            content_height += graph_rect.height + section_spacing
        
        # Update content height for scrolling
        self.stats_area.content_height = content_height + 20
    
    def render_population_stats(self):
        """Render population statistics"""
        # Set up variables for positioning
        content_height = 0
        line_height = 20
        section_spacing = 15
        
        # Population header
        pop_header = self.font_large.render("Population Statistics", True, COLOR_TEXT_HEADER)
        header_rect = self.stats_area.get_content_rect(content_height)
        if header_rect.top > 0 and header_rect.top < WINDOW_HEIGHT:
            self.screen.blit(pop_header, header_rect)
        content_height += line_height * 1.5
        
        # Basic population info
        pop_info = [
            f"Active Agents: {len(self.population.agents)}",
            f"Food Carried: {sum(1 for agent in self.population.agents.values() if agent.carrying)}",
            f"Total Food Stored: {sum(agent.store for agent in self.population.agents.values())}",
            f"Total Interactions: {len(self.population.interactions)}"
        ]
        
        for info in pop_info:
            info_rect = self.stats_area.get_content_rect(content_height)
            if info_rect.top > 0 and info_rect.top < WINDOW_HEIGHT:
                info_surf = self.font_normal.render(info, True, COLOR_TEXT)
                self.screen.blit(info_surf, info_rect)
            content_height += line_height
        
        # Population distribution visualization
        content_height += section_spacing * 2
        if self.visualization_cache["population_stats"] is not None:
            pop_rect = self.visualization_cache["population_stats"].get_rect(
                topleft=(self.stats_area.rect.left, 
                        self.stats_area.rect.top + content_height - self.stats_area.scroll_y)
            )
            
            if pop_rect.top + 20 > 0 and pop_rect.top < WINDOW_HEIGHT:
                self.screen.blit(self.visualization_cache["population_stats"], pop_rect)
            
            content_height += pop_rect.height + section_spacing
        
        # Agent list
        content_height += section_spacing
        agent_list_header = self.font_normal.render("Agent List", True, COLOR_TEXT_HEADER)
        agent_list_rect = self.stats_area.get_content_rect(content_height)
        if agent_list_rect.top > 0 and agent_list_rect.top < WINDOW_HEIGHT:
            self.screen.blit(agent_list_header, agent_list_rect)
        content_height += line_height * 1.5
        
        # Column headers
        col_headers = ["ID", "Energy", "Hunger", "Pain", "Food"]
        header_widths = [40, 60, 60, 60, 40]
        header_x = self.stats_area.rect.left
        
        for i, header in enumerate(col_headers):
            header_rect = pygame.Rect(
                header_x, 
                self.stats_area.rect.top + content_height - self.stats_area.scroll_y,
                header_widths[i], 
                line_height
            )
            
            if header_rect.top > 0 and header_rect.top < WINDOW_HEIGHT:
                header_surf = self.font_small.render(header, True, COLOR_TEXT)
                self.screen.blit(header_surf, header_surf.get_rect(midleft=header_rect.midleft))
            
            header_x += header_widths[i]
        
        content_height += line_height
        
        # Agent rows
        for agent_id, agent in self.population.agents.items():
            agent_num = agent_id.split('_')[1]
            row_data = [
                agent_num,
                f"{agent.energy:.0f}",
                f"{agent.hunger:.0f}",
                f"{agent.pain:.0f}",
                f"{agent.store}"
            ]
            
            row_x = self.stats_area.rect.left
            row_rect = pygame.Rect(
                row_x,
                self.stats_area.rect.top + content_height - self.stats_area.scroll_y,
                self.stats_area.rect.width,
                line_height
            )
            
            # Highlight selected agent
            if agent_id == self.selected_agent_id and row_rect.top > 0 and row_rect.top < WINDOW_HEIGHT:
                pygame.draw.rect(self.screen, (80, 100, 120), row_rect)
            
            # Draw row data
            for i, data in enumerate(row_data):
                cell_rect = pygame.Rect(
                    row_x,
                    row_rect.top,
                    header_widths[i],
                    line_height
                )
                
                if cell_rect.top > 0 and cell_rect.top < WINDOW_HEIGHT:
                    cell_surf = self.font_small.render(data, True, COLOR_TEXT)
                    self.screen.blit(cell_surf, cell_surf.get_rect(midleft=cell_rect.midleft))
                
                row_x += header_widths[i]
            
            content_height += line_height
        
        # Update content height for scrolling
        self.stats_area.content_height = content_height + 20
    
    def render_interaction_stats(self):
        """Render interaction statistics"""
        # Set up variables for positioning
        content_height = 0
        line_height = 20
        section_spacing = 15
        
        # Interaction header
        int_header = self.font_large.render("Interaction Statistics", True, COLOR_TEXT_HEADER)
        header_rect = self.stats_area.get_content_rect(content_height)
        if header_rect.top > 0 and header_rect.top < WINDOW_HEIGHT:
            self.screen.blit(int_header, header_rect)
        content_height += line_height * 1.5
        
        # Interaction summary
        if len(self.population.interactions) == 0:
            info_surf = self.font_normal.render("No interactions recorded yet", True, COLOR_TEXT)
            info_rect = self.stats_area.get_content_rect(content_height)
            if info_rect.top > 0 and info_rect.top < WINDOW_HEIGHT:
                self.screen.blit(info_surf, info_rect)
            content_height += line_height
        else:
            # Count interaction types
            interaction_types = {}
            for interaction in self.population.interactions:
                int_type = interaction["type"]
                if int_type not in interaction_types:
                    interaction_types[int_type] = 0
                interaction_types[int_type] += 1
            
            # Display interaction type counts
            for int_type, count in interaction_types.items():
                info = f"{int_type.capitalize()}: {count}"
                info_surf = self.font_normal.render(info, True, COLOR_TEXT)
                info_rect = self.stats_area.get_content_rect(content_height)
                if info_rect.top > 0 and info_rect.top < WINDOW_HEIGHT:
                    self.screen.blit(info_surf, info_rect)
                content_height += line_height
        
        # Interaction graph
        content_height += section_spacing * 2
        if self.visualization_cache["interaction_graph"] is not None:
            graph_rect = self.visualization_cache["interaction_graph"].get_rect(
                topleft=(self.stats_area.rect.left, 
                        self.stats_area.rect.top + content_height - self.stats_area.scroll_y)
            )
            
            if graph_rect.top + 20 > 0 and graph_rect.top < WINDOW_HEIGHT:
                self.screen.blit(self.visualization_cache["interaction_graph"], graph_rect)
            
            content_height += graph_rect.height + section_spacing
        
        # Recent interactions
        content_height += section_spacing
        recent_header = self.font_normal.render("Recent Interactions", True, COLOR_TEXT_HEADER)
        recent_rect = self.stats_area.get_content_rect(content_height)
        if recent_rect.top > 0 and recent_rect.top < WINDOW_HEIGHT:
            self.screen.blit(recent_header, recent_rect)
        content_height += line_height * 1.5
        
        # Show the 10 most recent interactions
        if len(self.population.interactions) > 0:
            recent = self.population.interactions[-10:]
            for interaction in reversed(recent):
                int_type = interaction["type"].capitalize()
                agents = [a.split('_')[1] for a in interaction["agents"]]
                info = f"{int_type} between Agent {agents[0]} and Agent {agents[1]}"
                
                info_surf = self.font_small.render(info, True, COLOR_TEXT)
                info_rect = self.stats_area.get_content_rect(content_height)
                if info_rect.top > 0 and info_rect.top < WINDOW_HEIGHT:
                    self.screen.blit(info_surf, info_rect)
                content_height += line_height
        
        # Update content height for scrolling
        self.stats_area.content_height = content_height + 20
    
    def run(self):
        """Main game loop"""
        while self.running:
            # Process events
            self.process_events()
            
            # Update state
            self.update()
            
            # Render
            self.render()
            
            # Cap the frame rate
            self.clock.tick(FPS)
        
        pygame.quit()
        logging.info("Simulation terminated")

# ───────────────────── Main Entry Point ─────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Emergence Simulation")
    parser.add_argument("--load", type=str, help="Path to a saved simulation to load")
    args = parser.parse_args()
    
    sim = Simulation()
    
    # Load saved simulation if specified
    if args.load and os.path.exists(args.load):
        world, population = load_simulation(args.load)
        if world and population:
            sim.world = world
            sim.population = population
            sim.selected_agent_id = list(sim.population.agents.keys())[0] if sim.population.agents else None
    
    # Run the simulation
    sim.run()
