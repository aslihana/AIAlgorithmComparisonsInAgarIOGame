import numpy as np

from models.ModelInterface import ModelInterface
import utils
import config as conf


class HeuristicModel(ModelInterface):
    def __init__(self):
        super().__init__()

    def get_action(self, state):
        (agents, foods, viruses, masses, time) = state
        
        # only get an action if agent is alive
        if self.id not in agents:
            return None
        my_agent = agents[self.id]

        # try to go towards/run away from nearest enemy first (if can consume/can be consumed)
        action = self.get_nearest_enemy_action(my_agent, agents)
        if action is None:
            # if no valid nearest enemy move, try to go towards nearest food
            action = self.get_nearest_food_action(my_agent, foods)

        return action

    def get_nearest_enemy_action(self, my_agent, agents):
        my_pos = my_agent.get_avg_pos()

        # find the nearest enemy object
        nearest_enemy = None
        nearest_enemy_dist = np.inf
        for enemy in agents.values():
            # don't include self in search for nearest enemy
            if enemy == my_agent:
                continue
            enemy_pos = enemy.get_avg_pos()
            curr_dist = utils.get_euclidean_dist(my_pos, enemy_pos)
            if curr_dist < nearest_enemy_dist:
                nearest_enemy = enemy
                nearest_enemy_dist = curr_dist

        # should only react to nearest enemy if it is sufficiently within agent's field of view
        if nearest_enemy is not None and nearest_enemy_dist < self.get_fov_dist():
            angle_to_enemy = utils.get_angle_between_points(my_pos, nearest_enemy.get_avg_pos())
            # if (likely) able to eat nearest enemy, go towards it
            if my_agent.get_avg_mass() > conf.CELL_CONSUME_MASS_FACTOR * nearest_enemy.get_avg_mass():
                return utils.get_action_closest_to_angle(angle_to_enemy)
            # if (likely) to be eaten by nearest enemy, run away from it
            elif nearest_enemy.get_avg_mass() > conf.CELL_CONSUME_MASS_FACTOR * my_agent.get_avg_mass():
                return utils.get_action_farthest_from_angle(angle_to_enemy)

        # if no nearest enemy or no consuming relationship, do nothing
        return None

    def get_nearest_food_action(self, my_agent, foods):
        my_pos = my_agent.get_avg_pos()
        my_rad = my_agent.get_avg_radius()

        # find the nearest food object reachable by the shortest path direction
        nearest_food = None
        nearest_food_dist = np.inf
        for food in foods:
            food_pos = food.get_pos()
            if self.is_pos_reachable(my_pos, my_rad, food_pos):
                curr_dist = utils.get_euclidean_dist(my_pos, food_pos)
                if curr_dist < nearest_food_dist:
                    nearest_food = food
                    nearest_food_dist = curr_dist

        # if there is a nearest food, get the direction that goes most directly to the nearest food object
        if nearest_food is not None:
            angle_to_food = utils.get_angle_between_points(
                my_pos, nearest_food.get_pos())
            return utils.get_action_closest_to_angle(angle_to_food)
        # otherwise, get a random action
        else:
            return utils.get_random_action()

    # considers FOV of agent to be circle with radius half smaller screen side length
    def get_fov_dist(self):
        return min(conf.SCREEN_WIDTH, conf.SCREEN_HEIGHT) / 2

    # helper function to determine if an agent can physically reach the given position
    def is_pos_reachable(self, my_pos, my_rad, other_pos):
        angle_to_other = utils.get_angle_between_points(my_pos, other_pos)
        action_to_angle = utils.get_action_closest_to_angle(angle_to_other)
        return utils.is_action_feasible(action_to_angle, my_pos, my_rad)

    # no optimization occurs for HeuristicModel
    def optimize(self):
        return

    # no remembering occurs for HeuristicModel
    def remember(self, state, action, next_state, reward, done):
        return