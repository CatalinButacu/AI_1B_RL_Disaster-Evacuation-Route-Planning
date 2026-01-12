import gymnasium as gym
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Set
import random
import sys
sys.path.insert(0, '..')

from env.hazard import HazardDynamics, SCENARIOS
import config


class EvacuationEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}
    
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
    ACTION_DELTAS = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}

    def __init__(self, grid_size: int = None, evacuation_zones: List[Tuple[int, int]] = None,
                 agent_start: Tuple[int, int] = None, initial_hazards: List[Tuple[int, int]] = None,
                 hazard_scenario: str = "flood", render_mode: str = None,
                 num_agents: int = None, num_exits: int = None, num_obstacles: int = None):
        super().__init__()
        
        self.grid_size = grid_size or config.GRID_SIZE
        self.num_agents = num_agents or getattr(config, 'NUM_AGENTS', 1)
        self.num_exits = num_exits or getattr(config, 'NUM_EXITS', 2)
        self.num_obstacles = num_obstacles or getattr(config, 'NUM_OBSTACLES', 0)
        self.render_mode = render_mode
        self.hazard_scenario_name = hazard_scenario
        
        self.obstacles: Set[Tuple[int, int]] = set()
        self.evacuation_zones: Set[Tuple[int, int]] = set()
        self.agent_starts: List[Tuple[int, int]] = []
        self.agent_positions: List[Tuple[int, int]] = []
        self.agent_active: List[bool] = []
        self.initial_hazards = initial_hazards or config.INITIAL_HAZARDS
        
        self._generate_environment(evacuation_zones, agent_start)
        
        self.observation_space = gym.spaces.Discrete(self.grid_size * self.grid_size)
        self.action_space = gym.spaces.Discrete(4)
        
        scenario_class = SCENARIOS.get(hazard_scenario, HazardDynamics)
        protected = list(self.evacuation_zones) + list(self.obstacles)
        self.hazard = scenario_class(grid_size=self.grid_size, initial_hazards=self.initial_hazards,
                                     protected_zones=protected)
        
        self.steps = 0
        self.agent_pos = self.agent_positions[0] if self.agent_positions else (0, 0)

    def _generate_environment(self, fixed_exits=None, fixed_start=None):
        # Use a local random generator for environment setup to ensure reproducibility
        # without affecting the global random state used for agent exploration.
        rng = random.Random(config.ENV_SEED)
        
        self.obstacles = set()
        center = self.grid_size // 2
        for _ in range(self.num_obstacles):
            for attempt in range(20):
                row = rng.randint(1, self.grid_size - 2)
                col = rng.randint(1, self.grid_size - 2)
                if abs(row - center) > 1 or abs(col - center) > 1:
                    if (row, col) not in self.obstacles:
                        self.obstacles.add((row, col))
                        break
        
        if fixed_exits:
            self.evacuation_zones = set(fixed_exits)
        else:
            self.evacuation_zones = set()
            far_edges = []
            for i in range(self.grid_size):
                far_edges.append((self.grid_size - 1, i))
                far_edges.append((i, self.grid_size - 1))
            far_edges = [e for e in far_edges if e not in self.obstacles]
            rng.shuffle(far_edges)
            for pos in far_edges[:self.num_exits]:
                self.evacuation_zones.add(pos)
        
        if fixed_start:
            self.agent_starts = [fixed_start]
        else:
            self.agent_starts = []
            available = []
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    pos = (row, col)
                    if pos not in self.obstacles and pos not in self.evacuation_zones:
                        available.append(pos)
            rng.shuffle(available)
            self.agent_starts = available[:self.num_agents]
        
        self.agent_positions = list(self.agent_starts)
        self.agent_active = [True] * len(self.agent_positions)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[int, Dict]:
        super().reset(seed=seed)
        
        protected = list(self.obstacles)
        self.hazard.protected_zones = set(protected)
        self.hazard.reset(self.initial_hazards)
        
        if not hasattr(self, '_env_initialized') or not self._env_initialized:
            self._place_exits_opposite_hazard()
            self._env_initialized = True
        
        self.agent_positions = list(self.agent_starts)
        self.agent_active = [True] * len(self.agent_starts)
        
        self.agent_pos = self.agent_positions[0] if self.agent_positions else (0, 0)
        self.hazard.protected_zones = set(list(self.evacuation_zones) + list(self.obstacles))
        self.steps = 0
        
        return self._pos_to_obs(self.agent_pos), self._get_info()
    
    def _place_exits_opposite_hazard(self):
        self.evacuation_zones = set()
        
        if hasattr(self.hazard, 'flood_origin'):
            origin_row, origin_col = self.hazard.flood_origin
        elif hasattr(self.hazard, 'fire_origin'):
            origin_row, origin_col = self.hazard.fire_origin
        elif hasattr(self.hazard, 'wave_centers') and self.hazard.wave_centers:
            # For earthquake, use the first wave center as the primary threat origin
            origin_row, origin_col = self.hazard.wave_centers[0]
        elif hasattr(self.hazard, 'tornado_pos'):
            # For tornado, use current position (though it moves)
            origin_row, origin_col = int(self.hazard.tornado_pos[0]), int(self.hazard.tornado_pos[1])
        else:
            origin_row, origin_col = 0, 0
        
        corners = [
            (0, 0), (0, self.grid_size - 1),
            (self.grid_size - 1, 0), (self.grid_size - 1, self.grid_size - 1)
        ]
        
        corners = sorted(corners, key=lambda c: abs(c[0] - origin_row) + abs(c[1] - origin_col), reverse=True)
        
        for corner in corners[:self.num_exits]:
            if corner not in self.obstacles:
                self.evacuation_zones.add(corner)
    
    def _place_agents_safe(self):
        self.agent_starts = []
        available = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                pos = (row, col)
                if pos not in self.obstacles and pos not in self.evacuation_zones:
                    available.append(pos)
        random.shuffle(available)
        self.agent_starts = available[:self.num_agents]
        self.agent_positions = list(self.agent_starts)
        self.agent_active = [True] * len(self.agent_positions)

    def step(self, action) -> Tuple[int, float, bool, bool, Dict]:
        self.steps += 1
        self.hazard.step()
        
        if isinstance(action, int):
            actions = [action] * len(self.agent_positions)
        else:
            actions = action
        
        total_reward = 0
        agent_rewards = [0.0] * len(self.agent_positions)
        
        occupied_cells = set(pos for pos, active in zip(self.agent_positions, self.agent_active) if active)
        
        for i, (pos, active) in enumerate(zip(self.agent_positions, self.agent_active)):
            if not active:
                continue
            
            old_pos = pos
            old_dist = self._distance_to_nearest_exit(old_pos)
            
            agent_action = actions[i] if i < len(actions) else 0
            d_row, d_col = self.ACTION_DELTAS[agent_action]
            new_row, new_col = pos[0] + d_row, pos[1] + d_col
            new_pos = (new_row, new_col)
            
            collision_enabled = getattr(config, 'COLLISION_AVOIDANCE', False)
            other_agents_at_target = new_pos in occupied_cells and new_pos != pos
            
            if not self._is_valid_pos(new_row, new_col):
                reward = config.REWARD_WALL
            elif new_pos in self.obstacles:
                reward = config.REWARD_OBSTACLE
            elif collision_enabled and other_agents_at_target:
                reward = getattr(config, 'REWARD_COLLISION', -2)
            else:
                occupied_cells.discard(pos)
                occupied_cells.add(new_pos)
                self.agent_positions[i] = new_pos
                reward = config.REWARD_STEP
                
                new_dist = self._distance_to_nearest_exit(new_pos)
                if new_dist < old_dist:
                    reward += getattr(config, 'REWARD_CLOSER_TO_EXIT', 2)
                elif new_dist > old_dist:
                    reward += getattr(config, 'REWARD_FARTHER_FROM_EXIT', -2)
            
            current_pos = self.agent_positions[i]
            
            if current_pos in self.evacuation_zones:
                reward = config.REWARD_EVACUATION
                self.agent_active[i] = False
            elif self.hazard.is_hazard(current_pos):
                reward = config.REWARD_HAZARD
                self.agent_active[i] = False
            
            agent_rewards[i] = reward
            total_reward += reward
        
        self.agent_pos = self.agent_positions[0]
        
        evacuated = sum(1 for i, active in enumerate(self.agent_active) 
                       if not active and self.agent_positions[i] in self.evacuation_zones)
        dead = sum(1 for i, active in enumerate(self.agent_active)
                  if not active and self.agent_positions[i] not in self.evacuation_zones)
        still_active = sum(self.agent_active)
        
        all_evacuated = evacuated == len(self.agent_positions)
        all_done = still_active == 0
        
        terminated = all_evacuated or all_done
        truncated = self.steps >= config.MAX_STEPS_PER_EPISODE
        
        info = self._get_info()
        info['agent_rewards'] = agent_rewards
        
        return self._pos_to_obs(self.agent_pos), total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode in ["ansi", "human"]:
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        grid = []
        for row in range(self.grid_size):
            row_str = ""
            for col in range(self.grid_size):
                pos = (row, col)
                if pos in self.agent_positions:
                    idx = self.agent_positions.index(pos)
                    row_str += f"{idx} " if self.agent_active[idx] else "x "
                elif pos in self.evacuation_zones:
                    row_str += "E "
                elif pos in self.obstacles:
                    row_str += "# "
                elif self.hazard.is_hazard(pos):
                    if self.hazard_scenario_name == 'tornado':
                        row_str += "🌀"
                    elif self.hazard_scenario_name == 'flood':
                        row_str += "~~"
                    elif self.hazard_scenario_name == 'fire':
                        row_str += "^^"
                    else:
                        row_str += "XX "
                else:
                    row_str += ". "
            grid.append(row_str)
        output = "\n".join(grid) + f"\nStep: {self.steps}"
        if self.render_mode == "human":
            print(output)
        return output

    def _pos_to_obs(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.grid_size + pos[1]

    def _distance_to_nearest_exit(self, pos: Tuple[int, int]) -> int:
        if not self.evacuation_zones:
            return self.grid_size * 2
        return min(abs(pos[0] - e[0]) + abs(pos[1] - e[1]) for e in self.evacuation_zones)

    def _is_valid_pos(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def _get_info(self) -> Dict:
        return {
            'agent_positions': self.agent_positions.copy(),
            'agent_active': self.agent_active.copy(),
            'agent_injured': getattr(self, 'agent_injured', [False]*len(self.agent_positions)),
            'hazard_positions': self.hazard.get_hazard_positions(),
            'obstacles': self.obstacles.copy(),
            'evacuation_zones': self.evacuation_zones.copy(),
            'steps': self.steps
        }

    @property
    def agent_start(self):
        return self.agent_starts[0] if self.agent_starts else (0, 0)


def make_evacuation_env(scenario: str = "flood", grid_size: int = None, 
                        render_mode: str = None, num_agents: int = None) -> EvacuationEnv:
    return EvacuationEnv(grid_size=grid_size, hazard_scenario=scenario, 
                         render_mode=render_mode, num_agents=num_agents)
