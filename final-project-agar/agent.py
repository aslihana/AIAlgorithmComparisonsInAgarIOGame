import pygame
import numpy as np
import config as conf
import utils
import math
import random
import statistics
from mass import Mass
from actions import Action

NORMAL_MODE = 'normal'
SHOOTING_MODE = 'shooting'


class AgentCell():
    def __init__(self, agent, x, y, radius=None, mass=None, mode=NORMAL_MODE):
        """
        An AgentCell is a single cell of an Agent

        Parameters:

            agent  (Agent)  : pointer to agent
            x      (number) : x position
            y      (number) : y position
            radius (number) : optional radius of the cell
            mass   (number) : mass of the cell
            mode   (string) : either NORMAL_MODE or SPLITTING_MODE
        """
        self.agent = agent
        self.x_pos = x
        self.y_pos = y

        self.mass = mass
        self.mode = mode

        self.is_alive = True

        if radius is not None:
            self.radius = radius
        else:
            self.radius = utils.mass_to_radius(mass)

    def get_velocity(self):
        # return int(max(conf.AGENT_STARTING_SPEED - (self.mass * 0.05), 1))
        if self.mass > 0:
            return max(utils.mass_to_velocity(self.mass), 1)
        else:
            return 1

    def add_mass(self, mass_to_add):
        self.set_mass(self.mass + mass_to_add)

    def set_mass(self, mass):
        """
        Setter method for the mass

        Also updates AgentCell radius

        Parameters

            mass (number)
        """
        if mass is None or mass <= 0:
            raise Exception('Mass must be positive')

        self.mass = mass
        self.radius = utils.mass_to_radius(mass)
        utils.move_in_bounds(self)

    def split(self):
        """
        Split this cell and return the newly created cell
        """
        self.set_mass(self.mass / 2)
        new_cell = AgentCell(self.agent, self.x_pos, self.y_pos,
                             self.radius, self.mass)
        return new_cell

    def eat_food(self, food):
        if food is None:
            raise ValueError('Cannot eat food which is None')
        self.add_mass(food.mass)

    def eat_mass(self, mass):
        if mass is None:
            raise ValueError('Cannot eat mass which is None')
        self.add_mass(mass.mass)

    def eat_virus(self, virus):
        """
        Have this `AgentCell` consume the provided `Virus`

        Increases cell mass, but causes the cell to split into as many cells as
        possible limited by:

        1. An `Agent` can have at most a certain number of cells

        2. An `AgentCell` needs to have at least a certain mass

        The above values are specified in the configuration file.

        Update the parent `Agent` object to house the state of all of the newly
        created cells.

        Parameters

            virus (Virus)

        Returns

            cells (AgentCell[]) : list of newly created cells
        """
        if virus is None:
            raise ValueError('Cannot eat virus which is None')

        self.add_mass(virus.mass)
        virus.is_alive = False

        # if the agent already has the max number of cells, a split can't happen
        if len(self.agent.cells) == conf.AGENT_CELL_LIMIT:
            return []

        max_cells_based_on_count = conf.AGENT_CELL_LIMIT - \
            len(self.agent.cells) + 1
        max_cells_based_on_size = int(self.mass / conf.MIN_CELL_MASS)
        num_cells_to_split_into = min(
            max_cells_based_on_count, max_cells_based_on_size)

        new_cells = []

        new_mass = self.mass / num_cells_to_split_into
        self.set_mass(new_mass)

        for _ in range(1, num_cells_to_split_into):
            new_cell = AgentCell(self.agent, self.x_pos,
                                 self.y_pos, mass=new_mass)
            new_cells.append(new_cell)

        self.agent.update_last_split()
        return new_cells

    def shoot(self, angle):
        self.mode = SHOOTING_MODE
        self.shooting_angle = angle
        self.shooting_velocity = self.radius
        self.shooting_acceleration = self.radius / 8

    def move_shoot(self):
        """
        Move in response to being shot
        """
        utils.move_object(self, self.shooting_angle, self.shooting_velocity)
        self.shooting_velocity = self.shooting_velocity - self.shooting_acceleration

        if self.shooting_velocity <= 0:
            # We are done being controlled by acceleration and can be controlled
            # by agent decisions

            # Change the mode
            self.mode = NORMAL_MODE

            # Clean out shooting state
            self.shooting_acceleration = None
            self.shooting_velocity = None
            self.shooting_acceleration = None

    def move(self, angle, vel):
        """
        Move in the direction specified by `angle` from the x axis in pos dir

        If `mode` is `shooting`, move behavior gets overriden

        Parameters

            angle (number) : between 0 and 360
            vel   (number) : can be positive or negative
        """
        if self.mode == SHOOTING_MODE:
            self.move_shoot()
        else:
            vel = vel if vel is not None else self.get_velocity()
            utils.move_object(self, angle, vel)

    def shift(self, dx=None, dy=None):
        """
        Adjust position by dx and dy

        NOTE does not check for collisions, borders, etc.

        Parameters

            dx (number)
            dy (number)
        """
        if dx is not None:
            self.x_pos += dx
        if dy is not None:
            self.y_pos += dy

    def get_pos(self):
        return (self.x_pos, self.y_pos)

    def handle_mass_decay(self):
        old_mass = self.mass
        new_mass = self.mass * conf.MASS_DECAY_FACTOR
        if new_mass < conf.MIN_CELL_MASS:
            return 0

        self.set_mass(new_mass)
        return (new_mass - old_mass)


class Agent():
    def __init__(self, game, model, x, y, radius, mass=None, color=None, name=None, manual_control=False, camera_follow=False):
        """
        An `Agent` is a player in the `Game`. An `Agent` can have many
        `AgentCells` (just one to start out with).

        Parameters

            game           (Game)      : game that this `Agent` belongs to
            model          (nn.Module) : the decision making model for this `Agent`
            x              (number)
            y              (number)
            radius         (number)
            mass           (number)
            color         
            name           (string)    : unique ID for the agent, displayed on the game
            manual_control (boolean)   : if should be controlled by user's keyboard
            camera_follow  (boolean)
        """
        self.game = game
        self.model = model

        self.color = color
        self.name = name
        self.angle = None  # For deciding direction to move in

        self.is_alive = True
        self.manual_control = manual_control

        # True if the game camera is following this agent
        self.camera_follow = camera_follow

        self.update_last_split()

        cell = AgentCell(self, x, y, radius=radius, mass=mass)
        self.cells = [cell]
        self.cells_lost = []

        self.starting_mass = mass
        self.max_mass = mass
        self.steps_taken = 0

    def handle_mass_decay(self):
        return sum([cell.handle_mass_decay() for cell in self.cells])
        # for cell in self.cells:
        #     cell.handle_mass_decay()

    def update_last_split(self):
        self.last_split = self.game.get_time()

    def do_action(self, action):
        self.steps_taken += 1
        self.max_mass = max(self.max_mass, self.get_mass())

        if action == Action.MOVE_RIGHT:
            self.angle = 0
        elif action == Action.MOVE_UP_RIGHT:
            self.angle = 45
        elif action == Action.MOVE_UP:
            self.angle = 90
        elif action == Action.MOVE_UP_LEFT:
            self.angle = 135
        elif action == Action.MOVE_LEFT:
            self.angle = 180
        elif action == Action.MOVE_DOWN_LEFT:
            self.angle = 225
        elif action == Action.MOVE_DOWN:
            self.angle = 270
        elif action == Action.MOVE_DOWN_RIGHT:
            self.angle = 315
        # elif action == Action.SPLIT:
        #     self.handle_split()
        else:
            raise ValueError('Agent received bad action in do_action()')

        self.move()

    def get_avg_x_pos(self):
        """
        @returns average x pos of all `AgentCells` belonging to this `Agent`
        """
        return sum([cell.x_pos for cell in self.cells]) / len(self.cells)

    def get_avg_y_pos(self):
        """
        @returns average y pos of all `AgentCells` belonging to this `Agent`
        """
        return sum([cell.y_pos for cell in self.cells]) / len(self.cells)

    def get_angle(self):
        return self.angle if self.angle is not None else 0

    def get_pos(self):
        """
        @returns tuple of average x and y pos of all `AgentCells` belonging to this `Agent`
        """
        return (self.get_avg_x_pos(), self.get_avg_y_pos())

    def get_avg_pos(self):
        return self.get_pos()

    def get_avg_radius(self):
        """
        @returns average radius of all `AgentCells` belonging to this `Agent`
        """
        return sum([cell.radius for cell in self.cells]) / len(self.cells)

    def get_avg_mass(self):
        return self.get_mass() / len(self.cells)

    def get_stdev_mass(self):
        if len(self.cells) < 2:
            return 0
        return statistics.stdev([cell.mass for cell in self.cells])

    def get_mass(self):
        """
        @returns summed mass of all `AgentCells` belonging to this `Agent`
        """
        return sum([cell.mass for cell in self.cells])

    def move(self, vel=None):
        if self.angle is None:
            return

        avg_x = self.get_avg_x_pos()
        avg_y = self.get_avg_y_pos()

        if len(self.cells) > 1:
            for cell in self.cells:
                # Handle converging towards the middle
                penalty = -2  # Move this many pixels towards the center
                angle_to_avg = utils.get_angle_between_points(
                    (avg_x, avg_y), cell.get_pos())

                if angle_to_avg is not None:
                    cell.move(angle_to_avg, penalty)

        for (idx, cell) in enumerate(self.cells):
            # Handle overlapping cells
            for otherIdx in range(idx + 1, len(self.cells)):
                otherCell = self.cells[otherIdx]
                overlap = utils.get_object_overlap(cell, otherCell)
                if overlap < 0:
                    continue
                dist_to_move = overlap / 2
                angle = utils.get_angle_between_objects(cell, otherCell)
                if angle is None:
                    # If they totally overlap with each other
                    angle = random.randrange(360)

                cell.move(angle, -1 * dist_to_move)
                otherCell.move(angle, dist_to_move)

            # Handle normal movement
            cell.move(self.angle, vel)

        # if the game camera is following this agent, pan it
        if self.camera_follow:
            self.game.camera.pan(self.get_avg_x_pos(), self.get_avg_y_pos())

    def handle_move_keys(self, keys, camera):
        is_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
        is_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
        is_up = keys[pygame.K_UP] or keys[pygame.K_w]
        is_down = keys[pygame.K_DOWN] or keys[pygame.K_s]

        # remove contradictory keys
        if is_left and is_right:
            is_left = False
            is_right = False

        if is_down and is_up:
            is_down = False
            is_up = False

        # perform appropriate action for key press
        if is_up:
            if is_left:
                self.do_action(Action.MOVE_UP_LEFT)
            elif is_right:
                self.do_action(Action.MOVE_UP_RIGHT)
            else:
                self.do_action(Action.MOVE_UP)
        elif is_down:
            if is_left:
                self.do_action(Action.MOVE_DOWN_LEFT)
            elif is_right:
                self.do_action(Action.MOVE_DOWN_RIGHT)
            else:
                self.do_action(Action.MOVE_DOWN)
        elif is_left:
            self.do_action(Action.MOVE_LEFT)
        elif is_right:
            self.do_action(Action.MOVE_RIGHT)

        self.move()

    def handle_shoot(self):
        # You can only shoot if you are a single cell
        if len(self.cells) > 1:
            return

        if self.get_mass() < conf.MIN_MASS_TO_SHOOT:
            return

        # Must be moving in a direction in order to shoot
        if self.angle is None:
            return

        cell = self.cells[0]
        cell.mass = cell.mass - conf.MASS_MASS

        (mass_x, mass_y) = cell.get_pos()
        mass = Mass(mass_x, mass_y, self.color, self.angle, cell.radius)
        self.game.add_mass(mass)

    def handle_merge(self):
        if len(self.cells) <= 1:
            return

        curr_time = self.game.get_time()
        if curr_time < self.last_split + conf.AGENT_TICKS_TO_MERGE_CELLS:
            return

        self.update_last_split()

        # Merge pairs of cells in the body
        merged_cells = []
        mid_idx = int(len(self.cells) / 2)
        for idx in range(0, mid_idx):
            cell = self.cells[idx]
            other_cell = self.cells[idx + mid_idx]
            avg_x_pos = (cell.x_pos + other_cell.x_pos) / 2
            avg_y_pos = (cell.y_pos + other_cell.y_pos) / 2
            merged_mass = cell.mass + other_cell.mass
            merged_cell = AgentCell(
                self, avg_x_pos, avg_y_pos, radius=None, mass=merged_mass)
            merged_cells.append(merged_cell)

        if len(self.cells) % 2 == 1:
            # Append last cell if there are an odd number
            merged_cells.append(self.cells[-1])

        self.cells = merged_cells

    def handle_split(self):
        print('[AGENT] handle split')
        if self.angle is None:
            return
        if len(self.cells) * 2 >= conf.AGENT_CELL_LIMIT:
            # Limit the nubmer of cells that an agent can be in
            return

        curr_time = self.game.get_time()
        if curr_time < self.last_split + conf.AGENT_TICKS_TO_SPLIT_AGAIN:
            return

        for cell in self.cells:
            # Each cell needs to be at least a certain size in order to split
            if (cell.mass / 2) < conf.MIN_CELL_MASS:
                return

        new_cells = []
        for cell in self.cells:
            new_cell = cell.split()
            new_cell.shoot(self.angle)
            new_cells.append(new_cell)

        self.cells = self.cells + new_cells
        self.update_last_split()

    def handle_other_keys(self, keys, camera):
        is_split = keys[pygame.K_SPACE]
        is_shoot = keys[pygame.K_q]

        if is_split:
            self.handle_split()
        elif is_shoot:
            self.handle_shoot()

    def act(self, state):
        action = self.model.get_action(state)
        self.do_action(action)
