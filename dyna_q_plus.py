import numpy as np
from collections import defaultdict
from typing import Tuple, Dict
import random
import sys
sys.path.insert(0, '..')

import config


class DynaQPlus:
    def __init__(self, n_states: int, n_actions: int, alpha: float = None, gamma: float = None,
                 epsilon: float = None, planning_steps: int = None, kappa: float = None):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha if alpha is not None else config.ALPHA
        self.gamma = gamma if gamma is not None else config.GAMMA
        self.epsilon = epsilon if epsilon is not None else config.EPSILON
        self.planning_steps = planning_steps if planning_steps is not None else config.PLANNING_STEPS
        self.kappa = kappa if kappa is not None else config.KAPPA
        
        self.Q = np.zeros((n_states, n_actions))
        self.model: Dict[Tuple[int, int], Tuple[float, int, bool]] = {}
        self.tau: Dict[Tuple[int, int], int] = defaultdict(int)
        self.time = 0

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_values = self.Q[state]
        return random.choice(np.nonzero(q_values == np.max(q_values))[0])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        
        self.model[(state, action)] = (reward, next_state, done)
        
        self.time += 1
        for key in self.tau:
            self.tau[key] += 1
        self.tau[(state, action)] = 0
        
        self._planning()

    def _planning(self):
        if not self.model:
            return
        for _ in range(self.planning_steps):
            s_plan, a_plan = random.choice(list(self.model.keys()))
            r_plan, s_next, done = self.model[(s_plan, a_plan)]
            bonus = self.kappa * np.sqrt(self.tau.get((s_plan, a_plan), 0))
            target = r_plan + bonus if done else r_plan + bonus + self.gamma * np.max(self.Q[s_next])
            self.Q[s_plan, a_plan] += self.alpha * (target - self.Q[s_plan, a_plan])

    def decay_epsilon(self, decay_rate: float = None, min_epsilon: float = None):
        decay = decay_rate if decay_rate is not None else config.EPSILON_DECAY
        min_eps = min_epsilon if min_epsilon is not None else config.EPSILON_MIN
        self.epsilon = max(min_eps, self.epsilon * decay)

    def get_policy(self) -> np.ndarray:
        return np.argmax(self.Q, axis=1)

    def get_value_function(self) -> np.ndarray:
        return np.max(self.Q, axis=1)

    def save(self, filepath: str):
        np.savez(filepath, Q=self.Q, model_keys=list(self.model.keys()),
                 model_values=list(self.model.values()), tau_keys=list(self.tau.keys()),
                 tau_values=list(self.tau.values()), time=self.time, epsilon=self.epsilon)

    def load(self, filepath: str):
        data = np.load(filepath, allow_pickle=True)
        self.Q = data['Q']
        self.model = dict(zip([tuple(k) for k in data['model_keys']],
                              [tuple(v) for v in data['model_values']]))
        self.tau = defaultdict(int, dict(zip([tuple(k) for k in data['tau_keys']], data['tau_values'])))
        self.time, self.epsilon = int(data['time']), float(data['epsilon'])


class StandardDynaQ(DynaQPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kappa = 0.0


class QLearning(DynaQPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_steps = 0
        self.kappa = 0.0


class SARSA(DynaQPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_steps = 0
        self.kappa = 0.0
        self.last_action = None

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        next_action = self.select_action(next_state) if not done else 0
        target = reward if done else reward + self.gamma * self.Q[next_state, next_action]
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


class ExpectedSARSA(DynaQPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_steps = 0
        self.kappa = 0.0

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        if done:
            target = reward
        else:
            q_next = self.Q[next_state]
            policy_probs = np.ones(self.n_actions) * self.epsilon / self.n_actions
            best_action = np.argmax(q_next)
            policy_probs[best_action] += 1.0 - self.epsilon
            expected_value = np.sum(policy_probs * q_next)
            target = reward + self.gamma * expected_value
        self.Q[state, action] += self.alpha * (target - self.Q[state, action])


class DoubleQLearning(DynaQPlus):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.planning_steps = 0
        self.kappa = 0.0
        self.Q2 = np.zeros((self.n_states, self.n_actions))

    def select_action(self, state: int, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        q_combined = self.Q[state] + self.Q2[state]
        return random.choice(np.nonzero(q_combined == np.max(q_combined))[0])

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        if random.random() < 0.5:
            best_action = np.argmax(self.Q[next_state])
            target = reward if done else reward + self.gamma * self.Q2[next_state, best_action]
            self.Q[state, action] += self.alpha * (target - self.Q[state, action])
        else:
            best_action = np.argmax(self.Q2[next_state])
            target = reward if done else reward + self.gamma * self.Q[next_state, best_action]
            self.Q2[state, action] += self.alpha * (target - self.Q2[state, action])

    def get_value_function(self) -> np.ndarray:
        return np.max(self.Q + self.Q2, axis=1) / 2


class PrioritizedSweeping(DynaQPlus):
    def __init__(self, *args, theta: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.theta = theta
        self.priority_queue = {}
        self.predecessors: Dict[int, set] = defaultdict(set)

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        priority = abs(target - self.Q[state, action])
        
        if priority > self.theta:
            self.priority_queue[(state, action)] = priority
        
        self.model[(state, action)] = (reward, next_state, done)
        self.predecessors[next_state].add((state, action))
        
        self._prioritized_planning()

    def _prioritized_planning(self):
        for _ in range(self.planning_steps):
            if not self.priority_queue:
                break
            
            (s, a) = max(self.priority_queue, key=self.priority_queue.get)
            del self.priority_queue[(s, a)]
            
            r, s_next, done = self.model[(s, a)]
            target = r if done else r + self.gamma * np.max(self.Q[s_next])
            self.Q[s, a] += self.alpha * (target - self.Q[s, a])
            
            self._update_predecessors(s)

    def _update_predecessors(self, s: int):
        for (s_bar, a_bar) in self.predecessors[s]:
            if (s_bar, a_bar) not in self.model:
                continue
            r_bar, _, done_bar = self.model[(s_bar, a_bar)]
            target_bar = r_bar if done_bar else r_bar + self.gamma * np.max(self.Q[s])
            priority = abs(target_bar - self.Q[s_bar, a_bar])
            if priority > self.theta:
                self.priority_queue[(s_bar, a_bar)] = priority


AGENTS = {
    'q_learning': QLearning,
    'sarsa': SARSA,
    'expected_sarsa': ExpectedSARSA,
    'double_q': DoubleQLearning,
    'dyna_q': StandardDynaQ,
    'dyna_q_plus': DynaQPlus,
    'prioritized_sweeping': PrioritizedSweeping
}
