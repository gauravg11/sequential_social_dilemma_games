"""Base class for an agent that defines the possible actions. """

from gym.spaces import Box, Discrete, Tuple, MultiDiscrete
import numpy as np
import utility_funcs as util

# basic moves every agent should do
BASE_ACTIONS = {0: 'MOVE_LEFT',  # Move left
                1: 'MOVE_RIGHT',  # Move right
                2: 'MOVE_UP',  # Move up
                3: 'MOVE_DOWN',  # Move down
                4: 'STAY',  # don't move
                5: 'TURN_CLOCKWISE',  # Rotate counter clockwise
                6: 'TURN_COUNTERCLOCKWISE'}  # Rotate clockwise

REDUCED_VOCAB_MAPPING = {0: 0, 1: 0, 2: 0, 3: 0,
                         4: 1,
                         5: 2, 6: 2,
                         7: 3,
                         8: 4}

MAP_follower = {'agent-0': .01, 'agent-1': .01, 'agent-2': .001, 'agent-3': .001, 'agent-4': .001}
MAP_leader = {'agent-0': .001, 'agent-1': .001, 'agent-2': .01, 'agent-3': .01, 'agent-4': .01}

class Agent(object):

    def __init__(self, agent_id, start_pos, start_orientation, grid, row_size, col_size, num_agents, num_symbols):
        """Superclass for all agents.

        Parameters
        ----------
        agent_id: (str)
            a unique id allowing the map to identify the agents
        start_pos: (np.ndarray)
            a 2d array indicating the x-y position of the agents
        start_orientation: (np.ndarray)
            a 2d array containing a unit vector indicating the agent direction
        grid: (2d array)
            a reference to this agent's view of the environment
        row_size: (int)
            how many rows up and down the agent can look
        col_size: (int)
            how many columns left and right the agent can look
        """
        self.agent_id = agent_id
        self.index = int(self.agent_id[-1])
        self.pos = np.array(start_pos)
        self.orientation = start_orientation
        # TODO(ev) change grid to env, this name is not very informative
        self.grid = grid
        self.row_size = row_size
        self.col_size = col_size
        self.num_agents = num_agents
        self.num_symbols = num_symbols
        self.reward_this_turn = 0

    @staticmethod
    def action_space(num_agents, num_symbols):
        """Identify the dimensions and bounds of the action space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete, or Tuple type
            a bounded box depicting the shape and bounds of the action space
        """
        raise NotImplementedError

    @staticmethod
    def observation_space(num_agents, num_symbols, view_len):
        """Identify the dimensions and bounds of the observation space.

        MUST BE implemented in new environments.

        Returns
        -------
        gym Box, Discrete or Tuple type
            a bounded box depicting the shape and bounds of the observation
            space
        """
        raise NotImplementedError

    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        raise NotImplementedError

    def get_state(self):
        return util.return_view(self.grid, self.get_pos(),
                                self.row_size, self.col_size)

    def compute_reward(self):
        reward = self.reward_this_turn
        self.reward_this_turn = 0
        return reward

    def compute_intrinsic_reward_reduced_vocab(self, game_action, prev_symbols, weight):
        prev_symbols = prev_symbols - 1
        if isinstance(game_action, list):
            # weight = MAP_leader[self.agent_id] # Used to define explicit leader-followers
            for i, action in enumerate(game_action):
                game_action[i] = REDUCED_VOCAB_MAPPING[action]
        else:
            # weight = MAP_follower[self.agent_id] # Used to define explicit leader-followers
            game_action = REDUCED_VOCAB_MAPPING[game_action]
        successes = np.count_nonzero(prev_symbols == game_action)
        return successes * weight

    def compute_intrinsic_reward(self, game_action, prev_symbols, weight):
        # symbol 0 maps to no_symbol, symbol 1 maps to action 1
        prev_symbols = prev_symbols - 1
        successes = np.count_nonzero(prev_symbols == game_action)
        return successes * weight

    # TODO: this is hacky and brittle
    def get_index(self):
        return self.index

    def set_pos(self, new_pos):
        self.pos = np.array(new_pos)

    def get_pos(self):
        return self.pos

    def translate_pos_to_egocentric_coord(self, pos):
        offset_pos = pos - self.get_pos()
        ego_centre = [self.row_size, self.col_size]
        return ego_centre + offset_pos

    def set_orientation(self, new_orientation):
        self.orientation = new_orientation

    def get_orientation(self):
        return self.orientation

    def get_map(self):
        return self.grid

    def return_valid_pos(self, new_pos):
        """Checks that the next pos is legal, if not return current pos"""
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        return temp_pos

    def update_agent_pos(self, new_pos):
        """Updates the agents internal positions

        Returns
        -------
        old_pos: (np.ndarray)
            2 element array describing where the agent used to be
        new_pos: (np.ndarray)
            2 element array describing the agent positions
        """
        old_pos = self.get_pos()
        ego_new_pos = new_pos  # self.translate_pos_to_egocentric_coord(new_pos)
        new_row, new_col = ego_new_pos
        # you can't walk through walls
        temp_pos = new_pos.copy()
        if self.grid[new_row, new_col] == '@':
            temp_pos = self.get_pos()
        self.set_pos(temp_pos)
        # TODO(ev) list array consistency
        return self.get_pos(), np.array(old_pos)

    def update_agent_rot(self, new_rot):
        self.set_orientation(new_rot)

    def hit(self, char):
        """Defines how an agent responds to being hit by a beam of type char"""
        raise NotImplementedError

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        raise NotImplementedError


HARVEST_ACTIONS = BASE_ACTIONS.copy()
HARVEST_ACTIONS.update({7: 'FIRE'})  # Fire a penalty beam

HARVEST_VIEW_SIZE = 7


class HarvestAgent(Agent):

    def __init__(self, agent_id, start_pos, start_orientation, grid, num_agents, num_symbols, view_len=HARVEST_VIEW_SIZE):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len, num_agents, num_symbols)
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @staticmethod
    def action_space(num_agents, num_symbols):
        action_dimension = [8]
        for i in range(num_agents):
            action_dimension.append(num_symbols)
        return MultiDiscrete(action_dimension)

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return HARVEST_ACTIONS[action_number]

    # @property
    @staticmethod
    def observation_space(num_agents, num_symbols, view_len=HARVEST_VIEW_SIZE):
        action_dimension = []
        for i in range(num_agents):
            action_dimension.append(num_symbols)
        return Tuple((Box(low=0.0, high=255.0, shape=(2 * view_len + 1,
                                             2 * view_len + 1, 3), dtype=np.float32)
                     ,
                     MultiDiscrete(action_dimension)))

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= 50

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 1

    def get_done(self):
        return False

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char


CLEANUP_ACTIONS = BASE_ACTIONS.copy()
CLEANUP_ACTIONS.update({7: 'FIRE',  # Fire a penalty beam
                        8: 'CLEAN'})  # Fire a cleaning beam

CLEANUP_VIEW_SIZE = 7


class CleanupAgent(Agent):
    def __init__(self, agent_id, start_pos, start_orientation, grid, num_agents, num_symbols, view_len=CLEANUP_VIEW_SIZE):
        self.view_len = view_len
        super().__init__(agent_id, start_pos, start_orientation, grid, view_len, view_len, num_agents, num_symbols)
        # remember what you've stepped on
        self.update_agent_pos(start_pos)
        self.update_agent_rot(start_orientation)

    @staticmethod
    def action_space(num_agents, num_symbols):
        action_dimension = [9]
        for i in range(num_agents):
            action_dimension.append(num_symbols)
        return MultiDiscrete(action_dimension)

    @staticmethod
    def observation_space(num_agents, num_symbols, view_len=CLEANUP_VIEW_SIZE):
        action_dimension = []
        for i in range(num_agents):
            action_dimension.append(num_symbols)
        return Tuple((Box(low=0.0, high=255.0, shape=(2 * view_len + 1,
                                             2 * view_len + 1, 3), dtype=np.float32)
                     ,
                     MultiDiscrete(action_dimension)))

    # Ugh, this is gross, this leads to the actions basically being
    # defined in two places
    def action_map(self, action_number):
        """Maps action_number to a desired action in the map"""
        return CLEANUP_ACTIONS[action_number]

    def fire_beam(self, char):
        if char == 'F':
            self.reward_this_turn -= 1

    def get_done(self):
        return False

    def hit(self, char):
        if char == 'F':
            self.reward_this_turn -= 50

    def consume(self, char):
        """Defines how an agent interacts with the char it is standing on"""
        if char == 'A':
            self.reward_this_turn += 1
            return ' '
        else:
            return char
