import gymnasium as gym
from mdp import RState

class MDPEnv(gym.Env):
    """
    Environment around an MDP which is given at initialization. The env contains the current state
    of the agent and will update it according to the actions. The agent is only able to observe the new state after its action.

    The initial position and the position after reset is given in the initializer.
    """

    def __init__(self, states: [RState], nr_actions: int, start_state_id: int):
        """
        Initialize the environment.
        Args:
            states:  list of states in the MDP
            nr_actions: the (maximal) number of actions for a state in the MDP
            start_state_id: id of the start state
        """
        # we keep a reference to the states for the interaction
        self._states = states

        # the start state and current state
        self._start_state_id = start_state_id
        self._current_state = self._states[self._start_state_id]

        # action space consists of moving in each direction
        self.action_space = gym.spaces.Discrete(nr_actions)

        # The observation is the id of the state after the action
        self.observation_space = gym.spaces.Discrete(len(states))

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        """

        Args:
            action: The action to take
        Returns:
            observation: the observation due to the agent's action
            reward: the reward as a result of the action
            terminated (bool): whether the agent reaches a terminal state
            truncated (bool): whether the episode is truncated due to the maximum number of steps
            info (dict): additional information
        """
        self._current_state, reward = self._current_state.take_action(action)
        return self._current_state.state_id, reward, self._current_state.is_terminal, False, {}

    def reset(self) -> tuple[int, dict]:
        """
        Reset the environment to the initial state.
        Returns:
            observation: the observation of the initial state
            info (dict): additional information
        """
        self._current_state = self._states[self._start_state_id]
        return self._current_state.state_id, {}

    def render(self, mode='human'):
        pass
