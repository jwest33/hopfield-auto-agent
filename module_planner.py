import random
import numpy as np
from typing import List, Tuple, Optional

# Constants pulled from agent module for convenience
MAX_H = 100
MAX_E = 100
MAX_P = 100
GRID = 40

class Goal:
    """Represents a persistent goal that an agent is pursuing"""
    
    def __init__(self, goal_type, target=None, priority=1.0, expiration=None):
        """
        Initialize a goal with a type, optional target, and priority.
        
        Args:
            goal_type (str): Type of goal ('find_food', 'return_home', 'store_food', 'rest', 'explore')
            target (tuple, optional): Target position if applicable
            priority (float): Goal priority (higher = more important)
            expiration (int, optional): Tick count when this goal expires
        """
        self.goal_type = goal_type
        self.target = target
        self.priority = priority
        self.expiration = expiration
        self.creation_time = None  # Will be set when added to agent
        self.plan = []  # List of actions to achieve this goal
        self.plan_index = 0  # Current position in the plan
        self.failed_attempts = 0  # Counter for failed attempts to achieve goal
    
    def is_expired(self, current_tick):
        """Check if the goal has expired"""
        if self.expiration is None:
            return False
        return current_tick >= self.expiration
    
    def next_action(self):
        """Get the next action in the plan"""
        if not self.plan or self.plan_index >= len(self.plan):
            return "REST"  # Default if no plan or at end of plan
        
        action = self.plan[self.plan_index]
        self.plan_index += 1
        return action
    
    def reset_plan(self):
        """Reset the plan execution index"""
        self.plan_index = 0
    
    def get_remaining_steps(self):
        """Get the number of remaining steps in the plan"""
        if not self.plan:
            return 0
        return max(0, len(self.plan) - self.plan_index)
    
    def __str__(self):
        """String representation of the goal"""
        if self.target:
            return f"{self.goal_type} at {self.target} (priority: {self.priority:.1f})"
        return f"{self.goal_type} (priority: {self.priority:.1f})"


class PlanningSystem:
    """Handles goal creation, prioritization, and planning for agents"""
    
    def __init__(self, agent):
        """Initialize the planning system for an agent"""
        self.agent = agent
        self.goals = []  # List of active goals
        self.current_goal = None  # Currently active goal
        self.tick_since_goal_change = 0  # Ticks since last goal change
        self.default_goal = Goal("explore", priority=0.1)  # Default goal when nothing else to do
        
        # Max number of goals to maintain
        self.max_goals = 5
        
        # Minimum ticks to stick with a goal before considering changing
        self.goal_persistence = 10
    
    def add_goal(self, goal):
        """Add a new goal to the system"""
        goal.creation_time = self.agent.tick_count
        
        # Check if we already have a similar goal
        for existing in self.goals:
            if existing.goal_type == goal.goal_type and existing.target == goal.target:
                # Update priority of existing goal if new one is higher priority
                if goal.priority > existing.priority:
                    existing.priority = goal.priority
                return
        
        # Add new goal and limit total goals
        self.goals.append(goal)
        self.goals.sort(key=lambda g: g.priority, reverse=True)
        
        if len(self.goals) > self.max_goals:
            self.goals = self.goals[:self.max_goals]
    
    def update_goal_priorities(self):
        """Update goals' priorities based on agent state with exponential urgency"""
        agent = self.agent
        
        # Calculate exponential urgency signals
        hunger_ratio = agent.hunger / MAX_H
        hunger_urgency = 1.0
        if hunger_ratio > 0.7:
            # Exponential increase for high hunger
            hunger_urgency = 1.0 + 2.0 * (np.exp(2 * (hunger_ratio - 0.7)) - 1)
        
        energy_ratio = agent.energy / MAX_E
        energy_urgency = 1.0
        if energy_ratio < 0.3:
            # Exponential increase for low energy
            energy_urgency = 1.0 + 2.0 * (np.exp(2 * (0.3 - energy_ratio)) - 1)
        
        pain_ratio = agent.pain / MAX_P
        pain_urgency = 1.0
        if pain_ratio > 0.6:
            # Exponential increase for high pain
            pain_urgency = 1.0 + 2.0 * (np.exp(2 * (pain_ratio - 0.6)) - 1)
        
        for goal in self.goals:
            # Increase food finding priority when hungry - now with exponential urgency
            if goal.goal_type == "find_food":
                goal.priority = max(goal.priority, hunger_ratio * 2.0 * hunger_urgency)
                
                # Extra priority boost if extremely hungry (survival instinct)
                if hunger_ratio > 0.9:
                    goal.priority = max(goal.priority, 4.0)
            
            # Increase return home priority with exponential urgency for low energy
            elif goal.goal_type == "return_home":
                if agent.carrying:
                    goal.priority = max(goal.priority, 1.5 * hunger_urgency)
                
                goal.priority = max(goal.priority, (1.0 - energy_ratio) * 2.0 * energy_urgency)
                
                # Extra priority boost if energy critically low
                if energy_ratio < 0.1:
                    goal.priority = max(goal.priority, 4.0)
            
            # Increase resting priority when in pain - with exponential urgency
            elif goal.goal_type == "rest":
                goal.priority = max(goal.priority, pain_ratio * 2.0 * pain_urgency)
                
                # Extra boost for critical pain
                if pain_ratio > 0.8:
                    goal.priority = max(goal.priority, 3.5)
            
            # Exploration priority decreases when urgent needs exist
            elif goal.goal_type == "explore":
                # Reduce exploration priority when urgent needs exist
                urgency_factor = max(hunger_urgency, energy_urgency, pain_urgency)
                if urgency_factor > 1.5:
                    goal.priority = max(0.05, goal.priority / urgency_factor)
                else:
                    # Exploration becomes more appealing if we haven't changed goals in a while
                    if self.tick_since_goal_change > 30:
                        goal.priority += 0.01
        
        # Reorder goals by updated priorities
        self.goals.sort(key=lambda g: g.priority, reverse=True)
        
    def should_change_goal(self):
        """Determine if the agent should change its current goal based on observations, needs and priors"""
        if not self.current_goal:
            return True
        
        agent = self.agent
        
        # Calculate utility scores for different goal types based on agent state
        goal_utilities = {
            "find_food": 0.0,
            "return_home": 0.0,
            "store_food": 0.0,
            "rest": 0.0,
            "explore": 0.0
        }
        
        # Hunger increases utility of food-related goals with sigmoid curve
        hunger_ratio = agent.hunger / MAX_H
        food_utility = 1.0 / (1.0 + np.exp(-10 * (hunger_ratio - 0.6)))  # Sigmoid centered at 60% hunger
        goal_utilities["find_food"] = food_utility * 2.0
        
        # Energy depletion increases return home utility
        energy_ratio = agent.energy / MAX_E
        home_utility = 1.0 / (1.0 + np.exp(-10 * (0.4 - energy_ratio)))  # Sigmoid centered at 40% energy
        goal_utilities["return_home"] = home_utility * 2.0
        
        # Pain increases rest utility
        pain_ratio = agent.pain / MAX_P
        rest_utility = 1.0 / (1.0 + np.exp(-10 * (pain_ratio - 0.5)))  # Sigmoid centered at 50% pain
        goal_utilities["rest"] = rest_utility * 1.5
        
        # Carrying food increases store_food utility
        if agent.carrying:
            goal_utilities["store_food"] = 1.5
        
        # Base exploration utility
        goal_utilities["explore"] = 0.2
        
        # Add time-based utility - more exploration if we've had the same goal for a while
        if self.tick_since_goal_change > 30:
            goal_utilities["explore"] += 0.1 * (self.tick_since_goal_change - 30) / 10
        
        # Adjust utilities based on environment observations
        current_cell = agent.w.cell(tuple(agent.pos))
        
        # If we're at home, slightly reduce the utility of returning home
        if current_cell.material == "home" or "home" in current_cell.tags:
            goal_utilities["return_home"] = 0
            # Being at home increases rest utility slightly
            goal_utilities["rest"] += 0.2
        
        # If food is visible, increase find_food utility
        if agent.check_nearby_food():
            goal_utilities["find_food"] += 0.5
        
        # If agent is in dangerous terrain, increase return_home utility
        if current_cell.local_risk > 0.3:
            goal_utilities["return_home"] += 0.3
        
        # Adjust based on weather conditions
        if agent.w.weather == "storm":
            goal_utilities["return_home"] += 0.3
            goal_utilities["explore"] -= 0.1
        
        # Check if current goal is still the best option
        current_goal_utility = goal_utilities.get(self.current_goal.goal_type, 0)
        
        # Factor in goal persistence - add bonus to current goal to prevent thrashing
        persistence_bonus = 0.3 * max(0, (self.goal_persistence - self.tick_since_goal_change)) / self.goal_persistence
        current_goal_utility += persistence_bonus
        
        # Find best alternative goal
        best_alternative_type = max(goal_utilities.items(), key=lambda x: x[1])[0]
        best_alternative_utility = goal_utilities[best_alternative_type]
        
        # Check if the goal has failed too many times
        if self.current_goal.failed_attempts > 5:
            return True
        
        # Check if goal is expired
        if self.current_goal.is_expired(agent.tick_count):
            return True
        
        # Decide based on utility difference
        # Require a significant utility difference to switch goals (helps with stability)
        utility_difference = best_alternative_utility - current_goal_utility
        utility_threshold = 0.3  # Minimum difference to switch
        
        # Important: More likely to switch if in critical state
        if hunger_ratio > 0.9 or energy_ratio < 0.1 or pain_ratio > 0.8:
            utility_threshold = 0.1  # Lower threshold for critical states
        
        return utility_difference > utility_threshold
    
    def select_best_goal(self):
        """Select the highest priority goal or default to exploration"""
        # Cleanup expired goals
        self.goals = [g for g in self.goals if not g.is_expired(self.agent.tick_count)]
        
        if not self.goals:
            return self.default_goal
        
        return self.goals[0]
    
    def validate_path(self, start_pos, actions):
        """
        Validate a sequence of actions against obstacles and return valid portion.
        
        Args:
            start_pos: Starting position [x, y]
            actions: List of actions ("N", "S", "E", "W", "REST")
            
        Returns:
            List of valid actions (stopping at first obstacle)
        """
        agent = self.agent
        world = agent.w
        valid_actions = []
        
        # Simulate movement along the path
        current_pos = start_pos.copy()
        for action in actions:
            if action == "REST" or action == "VERIFY_FOOD":
                valid_actions.append(action)
                continue
                
            # Get direction vector for this action
            dx, dy = agent.MOV[action]
            next_pos = [(current_pos[0] + dx) % GRID, (current_pos[1] + dy) % GRID]
            
            # Check if next position is passable
            next_cell = world.cell(tuple(next_pos))
            if next_cell.passable:
                valid_actions.append(action)
                current_pos = next_pos
            else:
                # Stop at first obstacle
                break
        
        return valid_actions

    def create_plan_for_goal(self, goal):
        """Create a sequence of actions to achieve the given goal"""
        agent = self.agent
        
        if goal.goal_type == "find_food":
            # Use the nearest food finder to locate food
            found, direction, distance = agent.find_nearest_food_direction()
            
            if found and direction:
                # We know where food is, make a plan to go there
                raw_plan = [direction] * min(distance, 15)  # Limit plan length
                
                # Validate the path against obstacles
                valid_plan = self.validate_path(agent.pos, raw_plan)
                
                # Add verification steps
                adapted_plan = []
                for i, action in enumerate(valid_plan):
                    adapted_plan.append(action)
                    # Every 5 steps, re-check if food is still visible
                    if (i+1) % 5 == 0 and i < len(valid_plan) - 1:
                        # This will cause the planning system to reassess at this point
                        adapted_plan.append("VERIFY_FOOD")
                
                goal.plan = adapted_plan
                goal.plan_index = 0
                goal.target = None  # We don't know exact coords, just direction
                return True
            else:
                # No food found, create exploration pattern
                # Use spiral or expanding square pattern to maximize coverage
                
                # Try to explore areas we haven't visited much
                least_visited_dirs = []
                
                # Create a mini exploration map (quadrants)
                visit_counts = {"N": 0, "S": 0, "E": 0, "W": 0}
                
                # Count visits to each cell type by direction
                if hasattr(agent, 'cell_experience'):
                    for cell_type, exp in agent.cell_experience.items():
                        if cell_type == "home" or cell_type == "food":
                            continue
                        
                        visits = exp.get("visits", 0)
                        # Associate cell types with directions based on material
                        if cell_type in ["dirt", "water"]:
                            visit_counts["N"] += visits
                            visit_counts["S"] += visits
                        elif cell_type in ["stone", "rock"]:
                            visit_counts["E"] += visits
                            visit_counts["W"] += visits
                
                # Sort directions by visit count (ascending)
                sorted_dirs = sorted(visit_counts.items(), key=lambda x: x[1])
                least_visited_dirs = [d[0] for d in sorted_dirs]
                
                # Use the least visited direction as primary
                main_dir = least_visited_dirs[0]
                
                # Create a spiral exploration pattern
                steps = [main_dir]
                secondary_dir = "E" if main_dir in ["N", "S"] else "N"
                
                # Spiral pattern: go N steps in one direction, then N steps in perpendicular direction,
                # then N+1 steps in opposite of first direction, etc.
                directions = [main_dir, secondary_dir, 
                             "S" if main_dir == "N" else ("N" if main_dir == "S" else main_dir),
                             "W" if secondary_dir == "E" else ("E" if secondary_dir == "W" else secondary_dir)]
                
                # Generate spiral by repeated applications of the pattern with increasing steps
                step_count = 2
                raw_plan = []
                for i in range(4):  # 4 iterations of the spiral
                    for j, direction in enumerate(directions):
                        steps_to_add = step_count if j % 2 == 0 else step_count
                        raw_plan.extend([direction] * steps_to_add)
                    step_count += 1
                
                # Validate the path against obstacles
                valid_plan = self.validate_path(agent.pos, raw_plan)
                
                # Add verification steps
                adapted_plan = []
                for i, action in enumerate(valid_plan):
                    adapted_plan.append(action)
                    # Every 5 steps, check if we found food
                    if (i+1) % 5 == 0 and i < len(valid_plan) - 1:
                        adapted_plan.append("VERIFY_FOOD")
                
                goal.plan = adapted_plan[:20]  # Limit plan length
                goal.plan_index = 0
                return True
        
        elif goal.goal_type == "return_home":
            # Calculate path to home
            home_x, home_y = agent.w.home
            curr_x, curr_y = agent.pos
            
            # Simple Manhattan path planning
            dx = home_x - curr_x
            dy = home_y - curr_y
            
            raw_plan = []
            # Add vertical movement steps
            if dx > 0:
                raw_plan.extend(["S"] * abs(dx))
            elif dx < 0:
                raw_plan.extend(["N"] * abs(dx))
                
            # Add horizontal movement steps
            if dy > 0:
                raw_plan.extend(["E"] * abs(dy))
            elif dy < 0:
                raw_plan.extend(["W"] * abs(dy))
            
            # If already at home, just plan to REST
            if not raw_plan:
                raw_plan = ["REST"]
            
            # IMPORTANT: Validate the path against obstacles
            valid_plan = self.validate_path(agent.pos, raw_plan)
            
            goal.plan = valid_plan
            goal.plan_index = 0
            goal.target = agent.w.home
            return True
        
        elif goal.goal_type == "rest":
            # Simple plan: just rest for a while
            goal.plan = ["REST"] * 5
            goal.plan_index = 0
            return True
        
        elif goal.goal_type == "store_food":
            # Calculate path to home
            home_x, home_y = agent.w.home
            curr_x, curr_y = agent.pos
            
            # Simple Manhattan path planning
            dx = home_x - curr_x
            dy = home_y - curr_y
            
            raw_plan = []
            # Add vertical movement steps
            if dx > 0:
                raw_plan.extend(["S"] * abs(dx))
            elif dx < 0:
                raw_plan.extend(["N"] * abs(dx))
                
            # Add horizontal movement steps
            if dy > 0:
                raw_plan.extend(["E"] * abs(dy))
            elif dy < 0:
                raw_plan.extend(["W"] * abs(dy))
            
            # If already at home, just plan to REST to store food
            if not raw_plan:
                raw_plan = ["REST"]
            else:
                # Add REST to store food after reaching home
                raw_plan.append("REST")
            
            # Validate the path against obstacles
            valid_plan = self.validate_path(agent.pos, raw_plan)
            
            goal.plan = valid_plan
            goal.plan_index = 0
            goal.target = agent.w.home
            return True
        
        elif goal.goal_type == "explore":
            # Create semi-random exploration pattern
            directions = ["N", "S", "E", "W"]
            
            # Find least visited areas from agent's experience
            least_visited = "dirt"  # Default
            min_visits = float('inf')
            
            for cell_type, exp in agent.cell_experience.items():
                if exp["visits"] < min_visits:
                    min_visits = exp["visits"]
                    least_visited = cell_type
            
            # Create a path that explores with some randomness
            raw_plan = []
            main_dir = random.choice(directions)
            
            for _ in range(15):
                if random.random() < 0.6:
                    raw_plan.append(main_dir)
                else:
                    raw_plan.append(random.choice(directions))
                
                # Occasionally switch direction to avoid going in circles
                if random.random() < 0.2:
                    main_dir = random.choice(directions)
            
            # Validate the path against obstacles
            valid_plan = self.validate_path(agent.pos, raw_plan)
            
            goal.plan = valid_plan
            goal.plan_index = 0
            return True
        
        return False
    
    def check_plan_progress(self):
        """Check if we're making progress on our current plan"""
        if not self.current_goal or not self.current_goal.plan:
            return False
        
        # If we're close to completing the plan, it's progressing well
        if self.current_goal.get_remaining_steps() < 3:
            return True
        
        # If the goal is to find food and we're carrying food, we succeeded
        if self.current_goal.goal_type == "find_food" and self.agent.carrying:
            return True
        
        # If goal is to reach home and we're there, we succeeded
        if (self.current_goal.goal_type == "return_home" or 
            self.current_goal.goal_type == "store_food"):
            
            agent = self.agent
            curr_cell = agent.w.cell(tuple(agent.pos))
            if curr_cell.material == "home" or "home" in curr_cell.tags:
                return True
        
        return False
    
    def repair_plan(self):
        """Attempt to repair the current plan if it's not working"""
        if not self.current_goal:
            return False
        
        # Increment failed attempts
        self.current_goal.failed_attempts += 1
        
        # Check if we can pass obstacles
        if self.current_goal.goal_type in ["find_food", "return_home", "store_food"]:
            agent = self.agent
            
            # Need to recalculate directions
            if self.current_goal.goal_type == "find_food":
                # Find food again
                found, direction, distance = agent.find_nearest_food_direction()
                if found and direction:
                    # New food direction found, update plan
                    raw_plan = [direction] * min(distance, 15)
                    valid_plan = self.validate_path(agent.pos, raw_plan)
                    self.current_goal.plan = valid_plan
                    self.current_goal.plan_index = 0
                    self.current_goal.failed_attempts = 0
                    return True
            
            elif self.current_goal.goal_type in ["return_home", "store_food"]:
                # Recalculate path home
                home_x, home_y = agent.w.home
                curr_x, curr_y = agent.pos
                
                # Try an alternate path - if we were going X then Y, now go Y then X
                dx = home_x - curr_x
                dy = home_y - curr_y
                
                # Build the opposite of the previous plan
                raw_plan = []
                
                # Horizontal first, then vertical
                if dy > 0:
                    raw_plan.extend(["E"] * abs(dy))
                elif dy < 0:
                    raw_plan.extend(["W"] * abs(dy))
                    
                if dx > 0:
                    raw_plan.extend(["S"] * abs(dx))
                elif dx < 0:
                    raw_plan.extend(["N"] * abs(dx))
                
                if not raw_plan:
                    raw_plan = ["REST"]
                elif self.current_goal.goal_type == "store_food":
                    # Add REST at the end for storing food
                    raw_plan.append("REST")
                
                # Validate the path against obstacles
                valid_plan = self.validate_path(agent.pos, raw_plan)
                
                self.current_goal.plan = valid_plan
                self.current_goal.plan_index = 0
                self.current_goal.failed_attempts = 0
                return True
        
        # Simpler goals like rest and explore can just reset their plan
        self.current_goal.reset_plan()
        return False
    
    def generate_goals_from_needs(self):
        """Generate appropriate goals based on agent needs"""
        agent = self.agent
        
        # Find food if hungry
        if agent.hunger > MAX_H * 0.5 and not agent.carrying:
            self.add_goal(Goal("find_food", priority=agent.hunger / MAX_H * 2.0))
        
        # Go home if carrying food
        if agent.carrying:
            self.add_goal(Goal("store_food", priority=1.5))
        
        # Return home if energy is low
        energy_deficit = 1.0 - (agent.energy / MAX_E)
        if energy_deficit > 0.6:
            self.add_goal(Goal("return_home", priority=energy_deficit * 1.8))
        
        # Rest if in pain or very low energy
        if agent.pain > MAX_P * 0.4 or agent.energy < MAX_E * 0.2:
            self.add_goal(Goal("rest", priority=1.5))
        
        # Always have a low-priority exploration goal
        if not any(g.goal_type == "explore" for g in self.goals):
            self.add_goal(Goal("explore", priority=0.1))
    
    def update(self):
        """Main update method for the planning system"""
        # Increment tick counter
        self.tick_since_goal_change += 1
        
        # Generate goals based on needs
        self.generate_goals_from_needs()
        
        # Update priorities based on current state
        self.update_goal_priorities()
        
        # Check if we need to change goals
        if self.should_change_goal():
            old_goal = self.current_goal
            self.current_goal = self.select_best_goal()
            
            # If we changed goals, reset the timer and create a new plan
            if old_goal != self.current_goal:
                self.tick_since_goal_change = 0
                if self.current_goal:
                    self.create_plan_for_goal(self.current_goal)
        
        # If no current goal, select one and create plan
        if not self.current_goal:
            self.current_goal = self.select_best_goal()
            self.create_plan_for_goal(self.current_goal)
            self.tick_since_goal_change = 0
        
        # Check if our plan is making progress
        progress = self.check_plan_progress()
        
        # Handle special verification actions
        if (self.current_goal and self.current_goal.plan and 
            self.current_goal.plan_index < len(self.current_goal.plan)):
            
            next_action = self.current_goal.plan[self.current_goal.plan_index]
            
            # Special "VERIFY_FOOD" action - recalculate food direction
            if next_action == "VERIFY_FOOD" and self.current_goal.goal_type == "find_food":
                self.current_goal.plan_index += 1  # Skip this special action
                
                # Recreate food plan with latest information
                self.create_plan_for_goal(self.current_goal)
                
                # Get the first action from the new plan
                if self.current_goal.plan and self.current_goal.plan_index < len(self.current_goal.plan):
                    return self.current_goal.plan[self.current_goal.plan_index]
                else:
                    return "REST"
        
        # If no progress or plan complete, try to repair or create new plan
        if not progress and self.current_goal.plan_index >= len(self.current_goal.plan):
            if not self.repair_plan():
                # If repair fails, recreate plan
                self.create_plan_for_goal(self.current_goal)
        
        # Return next action from current plan
        if self.current_goal and self.current_goal.plan:
            if self.current_goal.plan_index < len(self.current_goal.plan):
                action = self.current_goal.next_action()
                
                # Skip special actions (should be handled above)
                if action == "VERIFY_FOOD":
                    self.current_goal.plan_index += 1
                    if self.current_goal.plan_index < len(self.current_goal.plan):
                        action = self.current_goal.next_action()
                    else:
                        action = "REST"
                    
                return action
        
        # Fallback to rest
        return "REST"
