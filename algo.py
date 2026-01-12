import numpy as np
from collections import defaultdict
from typing import Tuple, Dict
import random
import config


class QLearning:
    def __init__(self, n_states: int, n_actions: int, alpha: float = None, 
                 gamma: float = None, epsilon: float = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha if alpha is not None else config.ALPHA
        self.gamma = gamma if gamma is not None else config.GAMMA
        self.epsilon = epsilon if epsilon is not None else config.EPSILON
        self.initial_epsilon = self.epsilon
        self.Q = np.zeros((n_states, n_actions))

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = self.Q[state]
        best_actions = np.nonzero(q_values == np.max(q_values))[0]
        return random.choice(best_actions)

    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        if done:
            target_value = reward
        else:
            target_value = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target_value - self.Q[state, action])

    def decay_epsilon(self, episode: int = None, total_episodes: int = None):
        if episode is not None and total_episodes is not None:
            # Linear decay based on episode percentage
            progress = episode / total_episodes
            self.epsilon = max(config.EPSILON_MIN, 
                               self.initial_epsilon - (self.initial_epsilon - config.EPSILON_MIN) * progress)
        else:
            # Fallback to exponential decay (legacy)
            decay = config.EPSILON_DECAY
            min_eps = config.EPSILON_MIN
            self.epsilon = max(min_eps, self.epsilon * decay)

    def save(self, filepath: str):
        np.savez(filepath, Q=self.Q, epsilon=self.epsilon)

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.Q = data['Q']
        self.epsilon = float(data['epsilon'])
        # Store initial_epsilon from current epsilon to maintain decay logic consistency after load
        self.initial_epsilon = self.epsilon


class DynaQ(QLearning):
    def __init__(self, n_states: int, n_actions: int, alpha: float = None, 
                 gamma: float = None, epsilon: float = None, 
                 planning_steps: int = None):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon)
        self.planning_steps = planning_steps if planning_steps is not None else config.PLANNING_STEPS
        self.environment_model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
        self.last_visit_time: Dict[Tuple[int, int], int] = {}
        self.visited_states: set = set()
        self.current_time_step = 0

    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool):
        super().update(state, action, reward, next_state, done)
        self.environment_model[(state, action)] = (reward, next_state, done)
        self.visited_states.add(state)
        self.current_time_step += 1
        self.last_visit_time[(state, action)] = self.current_time_step
        self.execute_planning_steps()

    def execute_planning_steps(self):
        if not self.environment_model:
            return
        for _ in range(self.planning_steps):
            sampled_state, sampled_action = random.choice(list(self.environment_model.keys()))
            sampled_reward, sampled_next_state, is_terminal = self.environment_model[(sampled_state, sampled_action)]
            if is_terminal:
                target_value = sampled_reward
            else:
                target_value = sampled_reward + self.gamma * np.max(self.Q[sampled_next_state])
            self.Q[sampled_state, sampled_action] += self.alpha * (target_value - self.Q[sampled_state, sampled_action])

    def save(self, filepath: str):
        np.savez(filepath, 
                 Q=self.Q, 
                 model_keys=list(self.environment_model.keys()),
                 model_values=list(self.environment_model.values()), 
                 last_visit_keys=list(self.last_visit_time.keys()),
                 last_visit_values=list(self.last_visit_time.values()), 
                 visited_states=list(self.visited_states),
                 time=self.current_time_step, 
                 epsilon=self.epsilon)

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.Q = data['Q']
        self.environment_model = dict(zip(
            [tuple(k) for k in data['model_keys']],
            [tuple(v) for v in data['model_values']]
        ))
        
        self.last_visit_time = dict(zip(
            [tuple(k) for k in data['last_visit_keys']], 
            data['last_visit_values']
        ))
        self.visited_states = set(data['visited_states'])
        self.current_time_step = int(data['time'])
        self.epsilon = float(data['epsilon'])
        self.initial_epsilon = self.epsilon


class DynaQPlus(DynaQ):
    def __init__(self, n_states: int, n_actions: int, alpha: float = None, 
                 gamma: float = None, epsilon: float = None, 
                 planning_steps: int = None, kappa: float = None):
        super().__init__(n_states, n_actions, alpha, gamma, epsilon, planning_steps)
        self.exploration_bonus_coefficient = kappa if kappa is not None else config.KAPPA

    def execute_planning_steps(self):
        if not self.visited_states:
            return
        
        visited_list = list(self.visited_states)
        for _ in range(self.planning_steps):
            sampled_state = random.choice(visited_list)
            # Standard Dyna-Q+ samples from ALL actions in a visited state
            sampled_action = random.randint(0, self.n_actions - 1)
            
            if (sampled_state, sampled_action) in self.environment_model:
                sampled_reward, sampled_next_state, is_terminal = self.environment_model[(sampled_state, sampled_action)]
            else:
                # Sutton & Barto: actions never tried are assumed to lead back to same state with 0 reward
                sampled_reward, sampled_next_state, is_terminal = 0.0, sampled_state, False
                
            last_visit = self.last_visit_time.get((sampled_state, sampled_action), 0)
            time_since_visit = self.current_time_step - last_visit
            exploration_bonus = self.exploration_bonus_coefficient * np.sqrt(time_since_visit)
            
            if is_terminal:
                target_value = sampled_reward + exploration_bonus
            else:
                target_value = sampled_reward + exploration_bonus + self.gamma * np.max(self.Q[sampled_next_state])
            self.Q[sampled_state, sampled_action] += self.alpha * (target_value - self.Q[sampled_state, sampled_action])


class RandomAgent:
    def __init__(self, n_states: int, n_actions: int, **kwargs):
        self.n_actions = n_actions
        self.epsilon = 1.0

    def select_action(self, state: int, greedy: bool = False) -> int:
        return random.randint(0, self.n_actions - 1)

    def update(self, *args, **kwargs):
        pass

    def decay_epsilon(self, *args, **kwargs):
        pass

    def save(self, filepath: str):
        np.savez(filepath, Q=np.zeros((1, 1)), epsilon=1.0)

    def load(self, filepath: str):
        pass


AGENTS = {
    'random': RandomAgent,
    'q_learning': QLearning,
    'dyna_q': DynaQ,
    'dyna_q_plus': DynaQPlus,
}
