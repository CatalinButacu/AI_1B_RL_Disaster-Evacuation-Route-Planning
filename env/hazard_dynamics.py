import random
from typing import Set, Tuple, List
import math
import sys
sys.path.insert(0, '..')

import config


class HazardDynamics:
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 spread_probability: float = None, max_spread_steps: int = None,
                 protected_zones: List[Tuple[int, int]] = None):
        self.grid_size = grid_size
        self.hazards: Set[Tuple[int, int]] = set(initial_hazards)
        self.initial_hazards = list(initial_hazards)
        self.spread_probability = spread_probability if spread_probability is not None else config.SPREAD_PROBABILITY
        self.max_spread_steps = max_spread_steps if max_spread_steps is not None else config.MAX_SPREAD_STEPS
        self.current_step = 0
        self.protected_zones = set(protected_zones) if protected_zones else set()

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        self.hazards = set(initial_hazards if initial_hazards else self.initial_hazards)
        self.current_step = 0

    def step(self) -> Set[Tuple[int, int]]:
        if self.current_step >= self.max_spread_steps:
            return set()
        self.current_step += 1
        return set()

    def is_hazard(self, position: Tuple[int, int]) -> bool:
        return position in self.hazards

    def get_hazard_positions(self) -> Set[Tuple[int, int]]:
        return self.hazards.copy()

    def is_stabilized(self) -> bool:
        return self.current_step >= self.max_spread_steps

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size


class FloodScenario(HazardDynamics):
    DIRECTIONS = {
        'NW': (0, 0),                    # Top-left corner
        'N': (0, None),                  # Top edge (center)
        'NE': (0, -1),                   # Top-right corner (use grid_size-1)
        'W': (None, 0),                  # Left edge (center)
        'E': (None, -1),                 # Right edge (center)
        'SW': (-1, 0),                   # Bottom-left corner
        'S': (-1, None),                 # Bottom edge (center)
        'SE': (-1, -1),                  # Bottom-right corner
    }
    
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 protected_zones: List[Tuple[int, int]] = None):
        super().__init__(grid_size, [], max_spread_steps=999999, protected_zones=protected_zones)
        self.flood_origin = (0, 0)
        self.flood_direction = 'NW'
        self.water_level = -3
        self.hazards = set()

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        super().reset([])
        self.flood_direction = 'NW'
        self.flood_origin = (0, 0)
        self.water_level = -3
        self.hazards = set()

    def step(self) -> Set[Tuple[int, int]]:
        self.water_level += 1
        new_hazards = set()
        
        if self.water_level > 0:
            origin_row, origin_col = self.flood_origin
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    dist = abs(row - origin_row) + abs(col - origin_col)
                    if dist <= self.water_level and (row, col) not in self.protected_zones:
                        if (row, col) not in self.hazards:
                            new_hazards.add((row, col))
                            self.hazards.add((row, col))
        
        self.current_step += 1
        return new_hazards
    
    def get_flood_direction(self) -> str:
        return self.flood_direction


class FireScenario(HazardDynamics):
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 protected_zones: List[Tuple[int, int]] = None):
        super().__init__(grid_size, [], max_spread_steps=grid_size * 2, protected_zones=protected_zones)
        self.fire_center = (grid_size // 2, grid_size // 2)
        self.fire_radius = -2
        self.hazards = set()

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        super().reset([])
        self.fire_center = (self.grid_size // 2, self.grid_size // 2)
        self.fire_radius = -2
        self.hazards = set()

    def step(self) -> Set[Tuple[int, int]]:
        if self.current_step >= self.max_spread_steps:
            return set()
        
        self.fire_radius += 0.4
        new_hazards = set()
        
        if self.fire_radius > 0:
            center_row, center_col = self.fire_center
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    dist = math.sqrt((row - center_row)**2 + (col - center_col)**2)
                    if dist <= self.fire_radius and (row, col) not in self.protected_zones:
                        if (row, col) not in self.hazards:
                            new_hazards.add((row, col))
                            self.hazards.add((row, col))
        
        self.current_step += 1
        return new_hazards


class EarthquakeScenario(HazardDynamics):
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 protected_zones: List[Tuple[int, int]] = None):
        super().__init__(grid_size, [], max_spread_steps=grid_size * 2, protected_zones=protected_zones)
        self.buildings = set()
        self.start_delay = 3

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        super().reset([])
        self.buildings = set()
        for _ in range(self.grid_size):
            r, c = random.randint(2, self.grid_size-2), random.randint(2, self.grid_size-2)
            if (r, c) not in self.protected_zones:
                self.buildings.add((r, c))
        self.hazards = set()
        self.start_delay = 3

    def step(self) -> Set[Tuple[int, int]]:
        if self.current_step >= self.max_spread_steps:
            return set()
        
        self.current_step += 1
        
        if self.start_delay > 0:
            self.start_delay -= 1
            return set()
        
        new_hazards = set()
        for building in list(self.buildings - self.hazards):
            if random.random() < 0.1:
                self.hazards.add(building)
                new_hazards.add(building)
        
        return new_hazards


class TornadoScenario(HazardDynamics):
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 protected_zones: List[Tuple[int, int]] = None):
        super().__init__(grid_size, [], max_spread_steps=grid_size * 3, protected_zones=protected_zones)
        self.tornado_pos = (0, 0)
        self.tornado_radius = 1
        self.rotation_angle = 0
        self.start_delay = 3

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        super().reset([])
        self.tornado_pos = (self.grid_size // 2, 0)
        self.rotation_angle = 0
        self.hazards = set()
        self.start_delay = 3

    def step(self) -> Set[Tuple[int, int]]:
        if self.current_step >= self.max_spread_steps:
            return set()
        
        self.current_step += 1
        
        if self.start_delay > 0:
            self.start_delay -= 1
            return set()
        
        self.rotation_angle += 30
        
        new_row = self.tornado_pos[0] + random.choice([-1, 0, 1])
        new_col = self.tornado_pos[1] + 1
        new_row = max(0, min(self.grid_size - 1, new_row))
        new_col = min(self.grid_size - 1, new_col)
        self.tornado_pos = (new_row, new_col)
        
        self.hazards = set()
        new_hazards = set()
        
        center_row, center_col = self.tornado_pos
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                dist = abs(row - center_row) + abs(col - center_col)
                if dist <= self.tornado_radius and (row, col) not in self.protected_zones:
                    self.hazards.add((row, col))
                    new_hazards.add((row, col))
        
        return new_hazards

    def get_tornado_center(self) -> Tuple[int, int]:
        return self.tornado_pos

    def get_rotation_angle(self) -> float:
        return self.rotation_angle


SCENARIOS = {
    'flood': FloodScenario,
    'fire': FireScenario,
    'earthquake': EarthquakeScenario,
    'tornado': TornadoScenario
}
