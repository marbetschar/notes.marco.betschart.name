import math
import numpy as np
import matplotlib.pyplot as plt

class RState:
    """
    State in a stochastic MDP. Each action can result in different states and rewards.
    """

    def __init__(self, state_id):
        self._state_id = state_id
        # each transition is tuple with (probability, next_state, reward)
        # for each action, we have a list of transitions
        self._transitions = {}
        self._is_terminal = False

        # we add a property `color` to the state for display
        self._color = None

    def __repr__(self):
        return f'State {self._state_id}'

    @property
    def state_id(self):
        return self._state_id

    @property
    def transitions(self):
        return self._transitions

    @property
    def is_terminal(self) -> bool:
        return self._is_terminal

    @is_terminal.setter
    def is_terminal(self, value):
        self._is_terminal = value

    @property
    def color(self):
        return self._color

    @color.setter
    def color(self, value):
        self._color = value

    def add_transition(self, action, next_state, reward, probability):
        """
        Add a transition to the state.
        Args:
            action: action for which the transition is added
            next_state: the next state after taking the action
            reward: the reward for taking the action
            probability: the probability of the transition
        """
        if action not in self._transitions:
            self._transitions[action] = []
        self._transitions[action].append([probability, next_state, reward])

    def add_transition_rescale(self, action, next_state, reward, probability):
        """
        Add a transition to the state with the give probability and rescale all other transition probabilities
        so that the probabilities sum to 1.0.

        Args:
            action: action for which the transition is added
            next_state: the next state after taking the action
            reward: the reward for taking the action
            probability: the probability of the transition
        """
        if action not in self._transitions:
            self._transitions[action] = []

        # check if the previous transitions sum to 1.0, if not they need to be rescaled for that too
        sum_prob = 0.0
        for p, _, _ in self._transitions[action]:
            sum_prob += p
        scale = 1.0 / sum_prob
        scale *= (1.0 - probability)
        transitions = self._transitions[action]
        for i in range(len(transitions)):
            transitions[i][0] *= scale
        self._transitions[action].append((probability, next_state, reward))

    def has_transition(self, action, target):
        """
        Return true if the state has a transition with the given action and target state.
        Args:
            action:
            target:

        Returns:
        """
        for p, next_state, reward in self._transitions[action]:
            if next_state == target:
                return True
        return False

    def remove_transition(self, action, target):
        """
        Remove a transition from the state. The first transition with the given target is removed.
        Args:
            action:
            target:
        Returns:
            the probability and reward of the removed transition
        """
        i = 0
        for p, next_state, reward in self._transitions[action]:
            if next_state == target:
                del self._transitions[action][i]
                return p, reward
            i += 1

    def take_action(self, action) -> (int, float):
        """
            Take an action in the state to get the next state and reward
        Args:
            action: the action to take

        Returns:
            the next state and reward
        """
        if self._is_terminal:
            raise Exception("Action on terminal state")

        # select a transition according to the probabilities
        transitions = self._transitions[action]
        probabilities = [p for p, _, _ in transitions]
        index = np.random.choice(len(transitions), p=probabilities)
        _, next_state, reward = transitions[index]
        return next_state, reward

    
                
        
    


class MDPGridworld:
    """
    MDP describing a gridworld.

    - States are (logically) distributed in a grid of size width x height
    - States are kept in 1D array by their state id
    - Actions are up, down, left, right (N, S, W, E)
    - Actions take you to the neighboring state per default, unless you are at the border
    - Action behaviour can be changed to be stochastic, or to transition to a different state

    - The origin of the coordinate system is in the upper left corner with x0 going down and x1 going right
    - This corresponds to the matrix indexing in numpy

    """
    N = 0
    E = 1
    S = 2
    W = 3
    NR_ACTIONS = 4

    def __init__(self, height, width):
        self._height = height
        self._width = width
        self._states = np.empty((height * width), dtype=object)

        # create the states first
        for x0 in range(self._height):
            for x1 in range(self._width):
                state_id = self.pos_to_id(x0, x1)
                s = RState(state_id)
                self._states[state_id] = s

        # create the transitions
        for x0 in range(self._height):
            for x1 in range(self._width):
                state_id = self.pos_to_id(x0, x1)
                s = self._states[state_id]

                # make a default grid with 4 actions for each state and connections to the neighbors
                # North
                if x0 != 0:
                    s.add_transition(self.N, self._states[self.pos_to_id(x0 - 1, x1)], -1.0, 1.0)
                else:
                    s.add_transition(self.N, self._states[self.pos_to_id(x0, x1)], -1.0, 1.0)

                # South
                if x0 != self._height - 1:
                    s.add_transition(self.S, self._states[self.pos_to_id(x0 + 1, x1)], -1.0, 1.0)
                else:
                    s.add_transition(self.S, self._states[self.pos_to_id(x0, x1)], -1.0, 1.0)
                # West
                if x1 != 0:
                    s.add_transition(self.W, self._states[self.pos_to_id(x0, x1 - 1)], -1.0, 1.0)
                else:
                    s.add_transition(self.W, self._states[self.pos_to_id(x0, x1)], -1.0, 1.0)

                # East
                if x1 != self._width - 1:
                    s.add_transition(self.E, self._states[self.pos_to_id(x0, x1 + 1)], -1.0, 1.0)
                else:
                    s.add_transition(self.E, self._states[self.pos_to_id(x0, x1)], -1.0, 1.0)

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @property
    def size(self):
        return self._height * self._width

    @property
    def states(self):
        return self._states

    def state(self, x0, x1):
        return self._states[self.pos_to_id(x0, x1)]

    def pos_to_id(self, x0, x1) -> int:
        #return x1 * self._height + x0
        return x0 * self._width + x1

    def id_to_pos(self, id) -> (int, int):
        #return id % self._height, id // self._height
        return id // self._width, id % self._width

    def all_state_ids(self):
        ids = np.zeros((self._height, self._width), dtype=int)
        for x0 in range(self._height):
            for x1 in range(self._width):
                ids[x0, x1] = self.pos_to_id(x0, x1)
        return ids

    def take_action(self, x0, x1, action) -> (int, int, float):
        """
            Take an action in the grid to get the next position and reward.

            Actions can also be performed directly using the underlying state objects. This method is just a
            convenience when you want to use the positions instead.
        Args:
            x0: x0 coordinate of the state
            x1: x1 coordinate of the state
            action: the action to take

        Returns:
            the next state and reward
        """
        state, reward = self._states[self.pos_to_id(x0, x1)].take_action(action)
        x0, x1 = self.id_to_pos(state.state_id)
        return x0, x1, reward

    def render(self, values, ax):
        values_2d = values.reshape(self._height, self._width)
        ax.matshow(values_2d, cmap='Blues', vmin=0, vmax=0)
        for i in range(self._height):
            for j in range(self._width):
                c = values_2d[i,j]
                ax.text(j, i+0.35, f'{c:.2f}', va='center', ha='center')

                if self.state(i,j).is_terminal:
                    r = plt.Rectangle((j-0.5,i-0.5), 1, 1, hatch='/')
                    ax.add_patch(r)

                if not self.state(i,j).color is None:
                    r = plt.Rectangle((j-0.5,i-0.5), 1, 1, fill=True, color=self.state(i,j).color)
                    ax.add_patch(r)

                # we also add the patch without fill
                r = plt.Rectangle((j-0.5,i-0.5), 1, 1, fill=False)
                ax.add_patch(r)


class WalledGridworld(MDPGridworld):
    """
    Gridworld that allows to add interior walls. Adding a wall changes all current transitions on both sides of the
    wall to remain in the same state.
    """
    def add_wall(self, x0, x1, y0, y1):
        """
        Add a wall to the gridworld. This changes the transitions of all states on both sides of the wall to remain
        in the same state. There is no check if the coordinates are actually neighbors.
        Args:
            x0: the x0 coordinate of the first grid position
            x1: the x1 coordinate of the first grid position
            y0: the x0 coordinate of the second grid position
            y1: the x1 coordinate of the second grid position
        """
        state_x = self.state(x0, x1)
        state_y = self.state(y0, y1)

        for a in range(self.NR_ACTIONS):
            if state_x.has_transition(a, state_y):
                p, reward = state_x.remove_transition(a, state_y)
                state_x.add_transition(a, state_x, reward, p)

            if state_y.has_transition(a, state_x):
                p, reward = state_y.remove_transition(a, state_x)
                state_y.add_transition(a, state_y, reward, p)


class SlipperyGridworld(WalledGridworld):
    def add_slippery_patch(self, x0, x1, probability):
        """
        Add a patch of slippery ground to the gridworld. This means that the agent has a chance of slipping down
        one grid position on any of the actions taken
        Args:
            x0: the x0 coordinate of the patch
            x1: the x1 coordinate of the patch
            probability: the probability to slip and fall one grid position down (S)
        """
        if x0 != self._height - 1:
            state_id = self.pos_to_id(x0, x1)
            state = self._states[state_id]
            state.color = 'lightblue'
            state.add_transition_rescale(self.N, self._states[self.pos_to_id(x0 + 1, x1)], -1.0, probability)
            state.add_transition_rescale(self.E, self._states[self.pos_to_id(x0 + 1, x1)], -1.0, probability)
            state.add_transition_rescale(self.W, self._states[self.pos_to_id(x0 + 1, x1)], -1.0, probability)


def policy_to_direction(p: np.array) -> (np.array, np.array):
    """
    Calculate the direction of the max value of the policy  into two arrays that can be plotted.
    """
    u = np.zeros((p.shape[0]), dtype=float)
    v = np.zeros((p.shape[0]), dtype=float)
    for i in range(p.shape[0]):
        direction = np.argmax(p[i, :])
        if p[i, direction] > 0:
            if direction == MDPGridworld.N:
                u[i] = 0.0
                v[i] = 1.0
            if direction == MDPGridworld.E:
                u[i] = 1.0
                v[i] = 0.0
            if direction == MDPGridworld.S:
                u[i] = 0.0
                v[i] = -1.0
            if direction == MDPGridworld.W:
                u[i] = -1.0
                v[i] = 0.0
    return u, v


def plot_policy_values(p, values, g):
    """
    Plot the policy as arrows, and the values as text using the render function from the gridworld g
    """
    fig, ax = plt.subplots()
    g.render(values, ax)

    u, v = policy_to_direction(p)
    u = u.reshape((g.height, g.width))
    v = v.reshape((g.height, g.width))
    ax.quiver(u, v)


def max_arg_with_ties(q):
    max_value = -math.inf
    tied_index = []

    for i in range(len(q)):
        if q[i] > max_value:
            max_value = q[i]
            tied_index = [i]
        elif q[i] == max_value:
            tied_index.append(i)
    return np.random.choice(tied_index)
