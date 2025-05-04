import random
import numpy as np
from typing import List, Tuple, Optional

# Constants pulled from agent module for convenience
MAX_H = 100
MAX_E = 100
MAX_P = 100
GRID = 40

def is_food_cell(cell):
    return (cell.material == "food" or "food" in cell.tags)

class Goal:
    """Represents a persistent goal that an agent is pursuing"""
    
    def __init__(self, goal_type, target=None, priority=1.0, expiration=None):
        """
        Initialize a goal with a type, optional target, and priority.
        
        Args:
            goal_type (str): Type of goal ('find_food', 'return_home', 'store_food', 'rest')
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
        self.default_goal = Goal("find_food", priority=0.1)  # Default goal when nothing else to do
        
        # Max number of goals to maintain
        self.max_goals = 5
        
        # Minimum ticks to stick with a goal before considering changing
        self.goal_persistence = 10
        
        # Path planning variables
        self.current_plan = []  # Current sequence of actions
        self.plan_index = 0  # Current position in the plan
        self.current_target = None  # Current target position
        self.consecutive_failures = 0  # Counter for consecutive plan failures
        
        # Initialize with a default goal
        self.add_goal(self.default_goal)
    
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
            "rest": 0.0
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
    
    def consider_path(self, path_segment):
        """
        Consider a path segment provided by another agent for potential navigation.
        
        Args:
            path_segment: List of positions forming a path segment
        """
        # Basic implementation - if path leads closer to home or current target, use it
        if not path_segment or len(path_segment) < 2:
            return  # Not enough information
            
        # Check if we have a current target
        if self.current_target:
            # Check if the path segment leads toward our target
            start_dist = abs(path_segment[0][0] - self.current_target[0]) + abs(path_segment[0][1] - self.current_target[1])
            end_dist = abs(path_segment[-1][0] - self.current_target[0]) + abs(path_segment[-1][1] - self.current_target[1])
            
            if end_dist < start_dist:
                # Path seems to lead toward our target, consider using it
                # We'll just take the last position as a waypoint
                self.set_target(path_segment[-1])
                return
        
        # If no target or path doesn't lead to target, check if it leads home
        home_x, home_y = self.agent.w.home
        start_home_dist = abs(path_segment[0][0] - home_x) + abs(path_segment[0][1] - home_y)
        end_home_dist = abs(path_segment[-1][0] - home_x) + abs(path_segment[-1][1] - home_y)
        
        if end_home_dist < start_home_dist:
            # Path leads closer to home, consider using it
            # Just take the last position as a waypoint
            self.set_target(path_segment[-1])

        
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
                # Find food again - note that find_nearest_food_direction now returns 4 values
                found, direction, distance, coordinates = agent.find_nearest_food_direction()
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
        
        # Simpler goals like rest can just reset their plan
        self.current_goal.reset_plan()
        return False
    
    def update(self):
        """Main update method for the planning system, returns next action"""
        # Increment tick counter
        self.tick_since_goal_change += 1
        
        # Generate goals based on needs
        self.generate_goals_from_needs()
        
        # Update priorities based on current state
        self.update_goal_priorities()
        
        # Get current agent state and check for food
        agent = self.agent
        current_cell = agent.w.cell(tuple(agent.pos))
        at_food_cell = is_food_cell(current_cell)
        
        # IMPORTANT: If the agent is at a food cell and not carrying food,
        # return REST to pick up the food (this forces an immediate pickup)
        if at_food_cell and not agent.carrying and self.current_goal and self.current_goal.goal_type == "find_food":
            print(f"Agent {agent.id}: At food cell! Resting to pick up food.")
            # This REST action will cause the agent's step function to pick up the food
            return "REST"
        
        # CRITICAL: If agent is carrying food and at home, REST to store it
        if agent.carrying and (current_cell.material == "home" or "home" in current_cell.tags):
            print(f"Agent {agent.id}: At home with food! Resting to store it.")
            return "REST"
        
        # Check if the agent just picked up food (transitioned from not carrying to carrying)
        if agent.carrying and self.current_goal and self.current_goal.goal_type == "find_food":
            # Successfully picked up food, now switch to store_food goal
            print(f"Agent {agent.id}: Successfully picked up food! Switching to store_food goal.")
            
            # Create and set a store_food goal
            store_goal = Goal("store_food", priority=2.0)
            self.add_goal(store_goal)
            self.current_goal = store_goal
            
            # IMPORTANT: Actually create a plan for this goal
            success = self.create_plan_for_goal(self.current_goal)
            if not success:
                print(f"Agent {agent.id}: Failed to create store_food plan! Creating fallback plan.")
                
                # Create fallback plan
                home_x, home_y = agent.w.home
                store_goal.target = [home_x, home_y]
                self.current_target = store_goal.target
                
                # Simple direct path to home
                dx = home_x - agent.pos[0]
                dy = home_y - agent.pos[1]
                
                plan = []
                if dx > 0:
                    plan.extend(["S"] * abs(dx))
                elif dx < 0:
                    plan.extend(["N"] * abs(dx))
                
                if dy > 0:
                    plan.extend(["E"] * abs(dy))
                elif dy < 0:
                    plan.extend(["W"] * abs(dy))
                
                # Add REST at the end to store food
                plan.append("REST")
                store_goal.plan = plan
                store_goal.plan_index = 0
            
            # Reset tick counter
            self.tick_since_goal_change = 0
            
            # Get first action of the new goal
            if self.current_goal.plan and self.current_goal.plan_index < len(self.current_goal.plan):
                print(f"Agent {agent.id}: First action in store_food plan: {self.current_goal.plan[self.current_goal.plan_index]}")
                return self.current_goal.plan[self.current_goal.plan_index]
            
            # Default action if plan creation failed
            return "REST"
        
        # Check if current goal's plan is complete
        if self.current_goal and self.current_goal.plan and self.current_goal.plan_index >= len(self.current_goal.plan):
            # Current plan is complete - immediately create a new plan or select a new goal
            # Print debug message
            print(f"Agent {agent.id}: Goal '{self.current_goal.goal_type}' plan completed. Evaluating completion.")
            
            # Check if the goal is actually complete
            goal_truly_complete = False
            
            # For food finding, only complete if agent is carrying food
            if self.current_goal.goal_type == "find_food":
                goal_truly_complete = agent.carrying
                
            # For returning home or storing food, check if we're at home
            elif self.current_goal.goal_type in ["return_home", "store_food"]:
                current_cell = agent.w.cell(tuple(agent.pos))
                goal_truly_complete = (current_cell.material == "home" or "home" in current_cell.tags)
                
            # For resting, always consider it complete when the plan is done
            elif self.current_goal.goal_type == "rest":
                goal_truly_complete = True
            
            if goal_truly_complete:
                # Goal is truly complete - trigger success handling
                print(f"Agent {agent.id}: Goal '{self.current_goal.goal_type}' truly complete! Triggering success handler.")
                self.plan_success()
            else:
                # The plan is complete but the goal isn't achieved
                # Try to repair the plan or create a new one
                if not self.repair_plan():
                    # Force goal reconsideration if repair fails
                    self.current_goal = None
                    self.tick_since_goal_change = 0
        
        # Check if we need to change goals (only if we still have a current goal)
        if self.current_goal and self.should_change_goal():
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
        
        # Handle special verification actions
        if (self.current_goal and self.current_goal.plan and 
            self.current_goal.plan_index < len(self.current_goal.plan)):
            
            next_action = self.current_goal.plan[self.current_goal.plan_index]
            
            # Special "VERIFY_FOOD" action - check if we've found food
            if next_action == "VERIFY_FOOD" and self.current_goal.goal_type == "find_food":
                self.current_goal.plan_index += 1  # Skip this special action
                
                # Check if we're at a food cell
                if at_food_cell:
                    # Found food! Return REST to pick it up
                    print(f"Agent {agent.id}: Found food during verification! Resting to pick it up.")
                    return "REST"
                else:
                    # No food found - recreate the plan with latest information
                    self.create_plan_for_goal(self.current_goal)
                    
                    # Get the first action from the new plan
                    if self.current_goal.plan and self.current_goal.plan_index < len(self.current_goal.plan):
                        # Update current plan from the current goal's plan
                        self.current_plan = self.current_goal.plan
                        self.plan_index = self.current_goal.plan_index
                        
                        return self.current_goal.plan[self.current_goal.plan_index]
                    else:
                        return "REST"
        
        # Update current plan from the current goal's plan
        if self.current_goal and self.current_goal.plan:
            self.current_plan = self.current_goal.plan
            self.plan_index = self.current_goal.plan_index
        
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
                
                # Make sure plan_index is synchronized
                self.plan_index = self.current_goal.plan_index
                    
                return action
        
        # Fallback to rest
        return "REST"
            
    def set_target(self, position):
        """
        Set the current target position for planning
        
        Args:
            position: [x, y] coordinates of the target
        """
        self.current_target = position
        # Reset consecutive failures when setting a new target
        self.consecutive_failures = 0
        
        # Force recalculation of plan for the new target
        self.repair_plan()

    def plan_success(self):
        """
        Mark the current plan as successful and update related metrics.
        Immediately transitions to the next appropriate goal.
        Properly handles food consumption or pickup based on agent needs.
        """
        self.consecutive_failures = 0
        # Get agent reference at the beginning to avoid UnboundLocalError
        agent = self.agent
        
        # Store current goal type before potentially changing goals
        current_goal_type = self.current_goal.goal_type if self.current_goal else None
        
        # Check current goal type to determine next appropriate action
        if current_goal_type == "find_food_to_eat":
            # Successfully found food to eat - consume it immediately
            print(f"Agent {agent.id}: Found food to eat! Consuming immediately")
            
            # Create a rest plan to consume food
            eat_goal = Goal("rest", priority=2.0)
            eat_goal.plan = ["REST"]
            eat_goal.plan_index = 0
            
            # The agent will consume food during the REST action
            self.add_goal(eat_goal)
            self.current_goal = eat_goal
            self.tick_since_goal_change = 0
            
        elif current_goal_type == "find_food_to_store":
            # Successfully found food - pick it up and take it home
            current_cell = agent.w.cell(tuple(agent.pos))
            at_food_cell = is_food_cell(current_cell)
            
            if at_food_cell and not agent.carrying:
                # We're at food but haven't picked it up yet
                print(f"Agent {agent.id}: Found food to store! Picking up to take home")
                
                # The agent should automatically pick up food when at food cell
                # But we need to create a plan to return home
                store_goal = Goal("store_food", priority=2.0)
                self.add_goal(store_goal)
                self.current_goal = store_goal
                self.create_plan_for_goal(self.current_goal)
            
            elif agent.carrying:
                # We've already picked up food, now go store it
                print(f"Agent {agent.id}: Successfully picked up food! Now returning home")
                
                # Create a high-priority store_food goal and make it active
                store_goal = Goal("store_food", priority=2.0)
                self.add_goal(store_goal)
                self.current_goal = store_goal
                # IMPORTANT: Actually create a plan to return home!
                success = self.create_plan_for_goal(self.current_goal)
                if not success:
                    print(f"Agent {agent.id}: ERROR - Failed to create plan to return home!")
                    # Fallback - direct path to home
                    home_x, home_y = agent.w.home
                    dx = home_x - agent.pos[0]
                    dy = home_y - agent.pos[1]
                    
                    # Create a simple direct plan as fallback
                    plan = []
                    if dx > 0:
                        plan.extend(["S"] * abs(dx))
                    elif dx < 0:
                        plan.extend(["N"] * abs(dx))
                    
                    if dy > 0:
                        plan.extend(["E"] * abs(dy))
                    elif dy < 0:
                        plan.extend(["W"] * abs(dy))
                    
                    plan.append("REST")  # Add REST to store food
                    self.current_goal.plan = plan
                    self.current_goal.plan_index = 0
                    
                print(f"Agent {agent.id}: Created store_food plan: {self.current_goal.plan}")
            
            # Reset tick counter
            self.tick_since_goal_change = 0
            
        elif current_goal_type == "store_food":
            # Successfully stored food, now should find more food or rest based on needs
            print(f"Agent {agent.id}: Food storage successful! Determining next urgent need")
            
            # CRITICAL: Verify that food was actually stored - check if still carrying
            if agent.carrying:
                print(f"Agent {agent.id}: ERROR - Agent still carrying food! Staying with store_food goal.")
                # Create a new plan to ensure food is stored
                store_goal = Goal("store_food", priority=2.0)
                store_goal.plan = ["REST"]  # Just rest to store the food
                store_goal.plan_index = 0
                self.add_goal(store_goal)
                self.current_goal = store_goal
                return
            
            # Check energy level and hunger to decide next actions
            energy_ratio = agent.energy / MAX_E
            hunger_ratio = agent.hunger / MAX_H
            
            # IMPORTANT: Create a new goal only if the food was actually stored
            if hunger_ratio > 0.7:
                # Very hungry - find food to eat
                print(f"Agent {agent.id}: Very hungry ({hunger_ratio:.2f}), finding food to eat")
                eat_goal = Goal("find_food_to_eat", priority=2.0)
                self.add_goal(eat_goal)
                self.current_goal = eat_goal
                self.create_plan_for_goal(self.current_goal)
            elif energy_ratio < 0.3:
                # Energy low, prioritize resting
                print(f"Agent {agent.id}: Energy low ({energy_ratio:.2f}), switching to rest goal")
                rest_goal = Goal("rest", priority=1.8)
                rest_goal.plan = ["REST"] * 3  # Rest for a few ticks
                rest_goal.plan_index = 0
                self.add_goal(rest_goal)
                self.current_goal = rest_goal
            else:
                # Energy OK and not very hungry, find more food to store
                print(f"Agent {agent.id}: Energy sufficient, finding more food to store")
                food_goal = Goal("find_food_to_store", priority=1.5)
                self.add_goal(food_goal)
                self.current_goal = food_goal
                self.create_plan_for_goal(self.current_goal)
            
            # Reset tick counter
            self.tick_since_goal_change = 0
        
        elif current_goal_type == "rest":
            # Successfully rested, now decide whether to eat or store food
            hunger_ratio = agent.hunger / MAX_H
            
            if hunger_ratio > 0.7:
                # Very hungry - prioritize finding food to eat
                print(f"Agent {agent.id}: After rest, hunger high ({hunger_ratio:.2f}), finding food to eat")
                eat_goal = Goal("find_food_to_eat", priority=2.0)
                self.add_goal(eat_goal)
                self.current_goal = eat_goal
                self.create_plan_for_goal(self.current_goal)
            else:
                # Not very hungry - prioritize finding food to store
                print(f"Agent {agent.id}: After rest, finding food to store")
                store_goal = Goal("find_food_to_store", priority=1.5)
                self.add_goal(store_goal)
                self.current_goal = store_goal
                self.create_plan_for_goal(self.current_goal)
            
            # Reset tick counter
            self.tick_since_goal_change = 0

    def create_plan_for_goal(self, goal):
        """Create a sequence of actions to achieve the given goal"""
        agent = self.agent
        
        # Find food to eat - can be any food anywhere
        if goal.goal_type == "find_food_to_eat":
            # Use the nearest food finder to locate food - now with explicit coordinates
            found, direction, distance, coordinates = agent.find_nearest_food_direction()
            
            if found and direction:
                if coordinates:
                    # We have exact coordinates, create a direct path to them
                    print(f"Agent {agent.id}: Planning path to food to eat at {coordinates}")
                    goal.target = coordinates
                    self.current_target = coordinates
                    
                    # Calculate simple path to the target (direct x-y movements)
                    path = []
                    current = agent.pos.copy()
                    
                    # First, handle x coordinate (using N/S)
                    dx = coordinates[0] - current[0]
                    if dx != 0:
                        # Handle wraparound
                        if abs(dx) > GRID // 2:
                            dx = -1 * (GRID - abs(dx)) * (1 if dx > 0 else -1)
                        
                        # Add N/S movements
                        if dx > 0:
                            path.extend(["S"] * abs(dx))
                        else:
                            path.extend(["N"] * abs(dx))
                    
                    # Now handle y coordinate (using E/W)
                    current[0] = coordinates[0]  # Update x to target
                    dy = coordinates[1] - current[1]
                    if dy != 0:
                        # Handle wraparound
                        if abs(dy) > GRID // 2:
                            dy = -1 * (GRID - abs(dy)) * (1 if dy > 0 else -1)
                        
                        # Add E/W movements
                        if dy > 0:
                            path.extend(["E"] * abs(dy))
                        else:
                            path.extend(["W"] * abs(dy))
                    
                    # Validate the path against obstacles
                    valid_path = self.validate_path(agent.pos, path)
                    
                    # Add REST to consume food
                    goal.plan = valid_path + ["REST"]
                    goal.plan_index = 0
                    
                    print(f"Agent {agent.id}: Created plan to food to eat: {goal.plan}")
                    return True
                else:
                    # No exact coordinates, but we have a direction and distance
                    # Handle adjacent food specially (distance = 1)
                    if distance == 1:
                        print(f"Agent {agent.id}: Food to eat is adjacent! Direct step in direction {direction}")
                        goal.plan = [direction, "REST"]
                        goal.plan_index = 0
                        
                        # Calculate target position for planning
                        dx, dy = agent.MOV[direction]
                        target_x = (agent.pos[0] + dx) % GRID
                        target_y = (agent.pos[1] + dy) % GRID
                        goal.target = [target_x, target_y]
                        self.current_target = goal.target
                        return True
                        
                    # For further food, create directional plan
                    print(f"Agent {agent.id}: Creating directional plan to food to eat: {direction} x {distance}")
                    raw_plan = [direction] * min(distance, 15)  # Limit plan length
                    
                    # Validate path against obstacles
                    valid_plan = self.validate_path(agent.pos, raw_plan)
                    
                    # Add REST at the end to consume food
                    goal.plan = valid_plan + ["REST"]
                    goal.plan_index = 0
                    
                    # Calculate the end position to set as target
                    target_pos = agent.pos.copy()
                    for act in valid_plan:
                        if act in agent.MOV:
                            dx, dy = agent.MOV[act]
                            target_pos[0] = (target_pos[0] + dx) % GRID
                            target_pos[1] = (target_pos[1] + dy) % GRID
                    
                    goal.target = target_pos.copy()
                    self.current_target = goal.target
                    
                    print(f"Agent {agent.id}: Created food to eat plan: {goal.plan} to target {goal.target}")
                    return True
            
            # No food found, create random search pattern
            print(f"Agent {agent.id}: No food to eat found, creating search pattern")
            directions = list(agent.MOV.keys())
            directions.remove("REST")  # Don't include REST in exploration
            
            plan = []
            for _ in range(10):  # Search for 10 steps
                plan.append(random.choice(directions))
                
            goal.plan = plan
            goal.plan_index = 0
            return True
        
        # Find food to store - should ignore food being carried by other agents
        elif goal.goal_type == "find_food_to_store":
            # Use the nearest food finder to locate food - now with explicit coordinates
            found, direction, distance, coordinates = agent.find_nearest_food_direction()
            
            # Skip food locations where other agents are carrying food
            if found and coordinates:
                # Check if any other agent is at this location and carrying food
                skip_location = False
                if hasattr(agent, 'w') and hasattr(agent.w, 'positions'):
                    for other_agent, pos in agent.w.positions.items():
                        if other_agent != agent and hasattr(other_agent, 'carrying'):
                            other_pos = list(pos) if isinstance(pos, tuple) else pos
                            if (other_pos[0] == coordinates[0] and 
                                other_pos[1] == coordinates[1] and 
                                other_agent.carrying):
                                skip_location = True
                                print(f"Agent {agent.id}: Skipping food at {coordinates} as another agent is carrying it")
                                break
                
                # Skip food at home
                home_x, home_y = agent.w.home
                if coordinates[0] == home_x and coordinates[1] == home_y:
                    skip_location = True
                    print(f"Agent {agent.id}: Skipping food at home location")
                
                if skip_location:
                    # Try to find another food location
                    # Simple approach: remove this food from consideration and search around agent
                    found = False
                    for radius in range(1, min(20, GRID // 2)):
                        for dx in range(-radius, radius + 1):
                            for dy in range(-radius, radius + 1):
                                if abs(dx) == radius or abs(dy) == radius:  # only check perimeter
                                    nx, ny = (agent.pos[0] + dx) % GRID, (agent.pos[1] + dy) % GRID
                                    # Skip the location we're avoiding
                                    if nx == coordinates[0] and ny == coordinates[1]:
                                        continue
                                        
                                    # Skip home location
                                    if nx == home_x and ny == home_y:
                                        continue
                                        
                                    # Skip locations where other agents with food
                                    skip_this = False
                                    if hasattr(agent, 'w') and hasattr(agent.w, 'positions'):
                                        for other_agent, pos in agent.w.positions.items():
                                            if other_agent != agent and hasattr(other_agent, 'carrying'):
                                                other_pos = list(pos) if isinstance(pos, tuple) else pos
                                                if (other_pos[0] == nx and 
                                                    other_pos[1] == ny and 
                                                    other_agent.carrying):
                                                    skip_this = True
                                                    break
                                    
                                    if skip_this:
                                        continue
                                    
                                    # Check for food at this location
                                    cell = agent.w.cell((nx, ny))
                                    if is_food_cell(cell):
                                        found = True
                                        coordinates = [nx, ny]
                                        # Calculate new direction and distance
                                        distance = abs(dx) + abs(dy)
                                        if abs(dx) > abs(dy):
                                            direction = "S" if dx > 0 else "N"
                                        else:
                                            direction = "E" if dy > 0 else "W"
                                        break
                            if found:
                                break
                        if found:
                            break
            
            if found and direction:
                if coordinates:
                    # We have exact coordinates, create a direct path to them
                    print(f"Agent {agent.id}: Planning path to food to store at {coordinates}")
                    goal.target = coordinates
                    self.current_target = coordinates
                    
                    # Calculate path as before
                    path = []
                    current = agent.pos.copy()
                    
                    # DEBUG: Print current position and target position
                    print(f"Agent {agent.id}: Current position: {current}, target: {coordinates}")
                    
                    # Calculate distance to verify it's not 0
                    real_distance = abs(coordinates[0] - current[0]) + abs(coordinates[1] - current[1])
                    print(f"Agent {agent.id}: Manhattan distance to target: {real_distance}")
                    
                    # CRITICAL: Check if we're already at the target location
                    if current[0] == coordinates[0] and current[1] == coordinates[1]:
                        print(f"Agent {agent.id}: Already at target location! Adding REST action")
                        goal.plan = ["REST"]
                        goal.plan_index = 0
                        return True
                    
                    # First, handle x coordinate (using N/S)
                    dx = coordinates[0] - current[0]
                    if dx != 0:
                        # Handle wraparound
                        if abs(dx) > GRID // 2:
                            dx = -1 * (GRID - abs(dx)) * (1 if dx > 0 else -1)
                        
                        # Add N/S movements
                        if dx > 0:
                            path.extend(["S"] * abs(dx))
                        else:
                            path.extend(["N"] * abs(dx))
                    
                    # Now handle y coordinate (using E/W)
                    current[0] = coordinates[0]  # Update x to target
                    dy = coordinates[1] - current[1]
                    if dy != 0:
                        # Handle wraparound
                        if abs(dy) > GRID // 2:
                            dy = -1 * (GRID - abs(dy)) * (1 if dy > 0 else -1)
                        
                        # Add E/W movements
                        if dy > 0:
                            path.extend(["E"] * abs(dy))
                        else:
                            path.extend(["W"] * abs(dy))
                    
                    # CRITICAL: Verify we actually created a path with movements
                    if not path:
                        print(f"Agent {agent.id}: WARNING - Empty path generated to target {coordinates}!")
                        # Try an alternative approach - add one step in each direction
                        # This helps break out of stuck situations
                        possible_dirs = ["N", "S", "E", "W"]
                        path = [random.choice(possible_dirs), random.choice(possible_dirs)]
                    
                    # Validate the path against obstacles
                    valid_path = self.validate_path(agent.pos, path)
                    
                    # CRITICAL: Verify valid path is not empty
                    if not valid_path:
                        print(f"Agent {agent.id}: WARNING - Valid path is empty! Adding random movement")
                        possible_dirs = ["N", "S", "E", "W"]
                        valid_path = [random.choice(possible_dirs)]
                    
                    # Add REST to pick up food
                    goal.plan = valid_path + ["REST"]
                    goal.plan_index = 0
                    
                    print(f"Agent {agent.id}: Created plan to food to store: {goal.plan}")
                    return True
                        
            # No suitable food found, create random search pattern
            print(f"Agent {agent.id}: No suitable food to store found, creating search pattern")
            directions = list(agent.MOV.keys())
            directions.remove("REST")  # Don't include REST
            
            plan = []
            for _ in range(10):  # Search for 10 steps
                plan.append(random.choice(directions))
                
            goal.plan = plan
            goal.plan_index = 0
            return True
        
        # Return home and store food goals
        elif goal.goal_type in ["return_home", "store_food"]:
            # Get home coordinates
            home_x, home_y = agent.w.home
            goal.target = [home_x, home_y]
            self.current_target = goal.target
            
            # Calculate path to home
            path = []
            current = agent.pos.copy()
            
            # Calculate distances considering wraparound
            dx = home_x - current[0]
            dy = home_y - current[1]
            
            # Handle wraparound for x coordinate
            if abs(dx) > GRID // 2:
                dx = -1 * (GRID - abs(dx)) * (1 if dx > 0 else -1)
                
            # Handle wraparound for y coordinate
            if abs(dy) > GRID // 2:
                dy = -1 * (GRID - abs(dy)) * (1 if dy > 0 else -1)
            
            # Create path - move in x direction first (N/S)
            if dx > 0:
                path.extend(["S"] * abs(dx))
            elif dx < 0:
                path.extend(["N"] * abs(dx))
            
            # Then move in y direction (E/W)
            if dy > 0:
                path.extend(["E"] * abs(dy))
            elif dy < 0:
                path.extend(["W"] * abs(dy))
            
            # If already at home, just add a REST
            if not path:
                path = ["REST"]
                
            # Validate the path against obstacles
            valid_path = self.validate_path(agent.pos, path)
            
            # Add REST at the end to store food (if store_food goal)
            if goal.goal_type == "store_food":
                valid_path.append("REST")
            
            goal.plan = valid_path
            goal.plan_index = 0
            
            print(f"Agent {agent.id}: Created plan to return home: {goal.plan}")
            return True
        
        elif goal.goal_type == "rest":
            # Simple rest plan - just rest in place
            goal.plan = ["REST"] * 3  # Rest for a few ticks
            goal.plan_index = 0
            return True
        
        # Failed to create a plan
        return False

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
        
        # Remove any old default "explore" goals
        self.goals = [g for g in self.goals if g.goal_type != "explore"]
        
        # Create different priority levels for eating vs storing food
        for goal in self.goals:
            # Increase food eating priority when very hungry - with exponential urgency
            if goal.goal_type == "find_food_to_eat":
                goal.priority = max(goal.priority, hunger_ratio * 2.5 * hunger_urgency)
                
                # Extra priority boost if extremely hungry (survival instinct)
                if hunger_ratio > 0.9:
                    goal.priority = max(goal.priority, 4.0)
            
            # Food storing has lower priority than eating when hungry
            elif goal.goal_type == "find_food_to_store":
                base_priority = 1.0
                # Lower priority if hungry
                if hunger_ratio > 0.6:
                    base_priority = 0.8
                # Lower priority if low energy
                if energy_ratio < 0.4:
                    base_priority = 0.6
                    
                goal.priority = max(goal.priority, base_priority)
                    
            # Increase return home priority with exponential urgency for low energy
            elif goal.goal_type == "return_home":
                if agent.carrying:
                    goal.priority = max(goal.priority, 1.5 * hunger_urgency)
                
                goal.priority = max(goal.priority, (1.0 - energy_ratio) * 2.0 * energy_urgency)
                
                # Extra priority boost if energy critically low
                if energy_ratio < 0.1:
                    goal.priority = max(goal.priority, 4.0)
            
            # Store food has high priority if carrying
            elif goal.goal_type == "store_food":
                if agent.carrying:
                    goal.priority = max(goal.priority, 2.0)
                else:
                    goal.priority = 0.0  # No priority if not carrying
            
            # Increase resting priority when in pain - with exponential urgency
            elif goal.goal_type == "rest":
                goal.priority = max(goal.priority, pain_ratio * 2.0 * pain_urgency)
                
                # Extra boost for critical pain
                if pain_ratio > 0.8:
                    goal.priority = max(goal.priority, 3.5)
                
                # Boost for low energy
                if energy_ratio < 0.2:
                    goal.priority = max(goal.priority, 3.0)
                    
        # Reorder goals by updated priorities
        self.goals.sort(key=lambda g: g.priority, reverse=True)

    def generate_goals_from_needs(self):
        """Generate appropriate goals based on agent needs"""
        agent = self.agent
        
        # Find food to eat if very hungry
        if agent.hunger > MAX_H * 0.7 and not agent.carrying:
            self.add_goal(Goal("find_food_to_eat", priority=agent.hunger / MAX_H * 2.5))
        
        # Find food to store if not very hungry and not carrying
        elif agent.hunger <= MAX_H * 0.6 and not agent.carrying:
            self.add_goal(Goal("find_food_to_store", priority=1.0))
        
        # Go home if carrying food
        if agent.carrying:
            self.add_goal(Goal("store_food", priority=2.0))
        
        # Return home if energy is low
        energy_deficit = 1.0 - (agent.energy / MAX_E)
        if energy_deficit > 0.6:
            self.add_goal(Goal("return_home", priority=energy_deficit * 1.8))
        
        # Rest if in pain or very low energy
        if agent.pain > MAX_P * 0.4 or agent.energy < MAX_E * 0.2:
            self.add_goal(Goal("rest", priority=1.5))
