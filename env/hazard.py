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

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self._in_bounds(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

class FloodScenario(HazardDynamics):
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 protected_zones: List[Tuple[int, int]] = None):
        super().__init__(grid_size, [], max_spread_steps=grid_size * 3, protected_zones=protected_zones)
        self.flood_origin = (0, 0)
        self.water_level = 0
        self.spread_rate = 0.6

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        super().reset([])
        self.flood_origin = (0, 0)
        self.spread_rate = 0.6
        self.water_level = 0
        self.hazards = set()

    def step(self) -> Set[Tuple[int, int]]:
        self.current_step += 1
        self.water_level += self.spread_rate
        
        new_hazards = set()
        
        if self.water_level > 0:
            origin_row, origin_col = self.flood_origin
            current_radius = int(self.water_level)
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if (row, col) in self.hazards or (row, col) in self.protected_zones:
                        continue
                    
                    dist = abs(row - origin_row) + abs(col - origin_col)
                    
                    if dist <= current_radius:
                        spread_prob = 1.0 - (dist / (current_radius + 1)) * 0.3
                        if random.random() < spread_prob:
                            new_hazards.add((row, col))
                            self.hazards.add((row, col))
        
        return new_hazards

class FireScenario(HazardDynamics):
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 protected_zones: List[Tuple[int, int]] = None):
        super().__init__(grid_size, [], max_spread_steps=grid_size * 3, protected_zones=protected_zones)
        self.fire_origin = (0, 0)
        self.wind_direction = (1, 1)
        self.base_spread_prob = 0.5
        self.ember_jump_prob = 0.08

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        super().reset([])
        self.fire_origin = (0, 0)
        self.wind_direction = (1, 1)
        self.hazards = {self.fire_origin}

    def step(self) -> Set[Tuple[int, int]]:
        if self.current_step >= self.max_spread_steps:
            return set()
        
        self.current_step += 1
        new_hazards = set()
        
        for fire_pos in list(self.hazards):
            for neighbor in self._get_neighbors(fire_pos):
                if neighbor in self.hazards or neighbor in self.protected_zones:
                    continue
                
                burning_neighbors = sum(1 for n in self._get_neighbors(neighbor) if n in self.hazards)
                
                dr = neighbor[0] - fire_pos[0]
                dc = neighbor[1] - fire_pos[1]
                wind_bonus = 0.15 if (dr, dc) == self.wind_direction else 0
                
                spread_prob = min(0.8, self.base_spread_prob + burning_neighbors * 0.1 + wind_bonus)
                
                if random.random() < spread_prob:
                    new_hazards.add(neighbor)
        
        if random.random() < self.ember_jump_prob and self.hazards:
            source = random.choice(list(self.hazards))
            jump_dist = random.randint(2, 4)
            ember_row = source[0] + self.wind_direction[0] * jump_dist + random.randint(-1, 1)
            ember_col = source[1] + self.wind_direction[1] * jump_dist + random.randint(-1, 1)
            
            if self._in_bounds(ember_row, ember_col):
                ember_pos = (ember_row, ember_col)
                if ember_pos not in self.hazards and ember_pos not in self.protected_zones:
                    new_hazards.add(ember_pos)
        
        self.hazards.update(new_hazards)
        return new_hazards

class EarthquakeScenario(HazardDynamics):
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 protected_zones: List[Tuple[int, int]] = None):
        super().__init__(grid_size, [], max_spread_steps=grid_size * 4, protected_zones=protected_zones)
        self.wave_centers: List[Tuple[int, int]] = []
        self.current_wave = 0
        self.wave_timer = 0
        self.wave_pause = 8
        self.shockwave_radii: List[float] = []
        self.max_waves = 4

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        super().reset([])
        self.wave_centers = [
            (0, 0),
            (self.grid_size // 4, self.grid_size // 4),
            (self.grid_size // 2, 0),
            (0, self.grid_size // 2)
        ]
        self.shockwave_radii = [0.0] * len(self.wave_centers)
        self.current_wave = 0
        self.wave_timer = 0
        self.hazards = set()

    def step(self) -> Set[Tuple[int, int]]:
        if self.current_step >= self.max_spread_steps:
            return set()
        
        self.current_step += 1
        new_hazards = set()
        
        for i in range(self.current_wave + 1):
            if i < len(self.shockwave_radii):
                self.shockwave_radii[i] += 0.6
        
        for i in range(self.current_wave + 1):
            if i >= len(self.wave_centers):
                continue
            center = self.wave_centers[i]
            radius = self.shockwave_radii[i]
            
            for row in range(self.grid_size):
                for col in range(self.grid_size):
                    if (row, col) in self.hazards or (row, col) in self.protected_zones:
                        continue
                    
                    dist = math.sqrt((row - center[0])**2 + (col - center[1])**2)
                    
                    if abs(dist - radius) < 1.5:
                        collapse_prob = 0.25
                        if random.random() < collapse_prob:
                            self.hazards.add((row, col))
                            new_hazards.add((row, col))
        
        self.wave_timer += 1
        if self.wave_timer >= self.wave_pause and self.current_wave < self.max_waves - 1:
            self.current_wave += 1
            self.wave_timer = 0
        
        return new_hazards

class TornadoScenario(HazardDynamics):
    def __init__(self, grid_size: int, initial_hazards: List[Tuple[int, int]],
                 protected_zones: List[Tuple[int, int]] = None):
        super().__init__(grid_size, [], max_spread_steps=grid_size * 4, protected_zones=protected_zones)
        self.tornado_pos = [0.0, 0.0]
        self.tornado_radius = 2
        self.velocity = [0.8, 0.8]

    def reset(self, initial_hazards: List[Tuple[int, int]] = None):
        super().reset([])
        self.tornado_pos = [0.0, 0.0]
        self.velocity = [0.8, 0.8]
        self.tornado_radius = 2
        self.hazards = set()

    def step(self) -> Set[Tuple[int, int]]:
        if self.current_step >= self.max_spread_steps:
            return set()
        
        self.current_step += 1
        self.velocity[0] += random.uniform(-0.2, 0.2)
        self.velocity[1] += random.uniform(-0.2, 0.2)
        
        speed = math.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        if speed > 1.5:
            self.velocity[0] *= 1.5 / speed
            self.velocity[1] *= 1.5 / speed
        if speed < 0.5:
            self.velocity[0] *= 0.8 / max(0.1, speed)
            self.velocity[1] *= 0.8 / max(0.1, speed)
        
        if random.random() < 0.05:
            self.velocity[0] += random.uniform(-0.8, 0.8)
            self.velocity[1] += random.uniform(-0.8, 0.8)
        
        self.tornado_pos[0] += self.velocity[0]
        self.tornado_pos[1] += self.velocity[1]
        
        if self.tornado_pos[0] < 0 or self.tornado_pos[0] >= self.grid_size:
            self.velocity[0] *= -1
            self.tornado_pos[0] = max(0, min(self.grid_size - 1, self.tornado_pos[0]))
        if self.tornado_pos[1] < 0 or self.tornado_pos[1] >= self.grid_size:
            self.velocity[1] *= -1
            self.tornado_pos[1] = max(0, min(self.grid_size - 1, self.tornado_pos[1]))
        
        self.hazards = set()
        new_hazards = set()
        
        center_row, center_col = int(self.tornado_pos[0]), int(self.tornado_pos[1])
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                dist = math.sqrt((row - center_row)**2 + (col - center_col)**2)
                if dist <= self.tornado_radius and (row, col) not in self.protected_zones:
                    self.hazards.add((row, col))
                    new_hazards.add((row, col))
        
        return new_hazards

    def get_tornado_center(self) -> Tuple[int, int]:
        return (int(self.tornado_pos[0]), int(self.tornado_pos[1]))

SCENARIOS = {
    'flood': FloodScenario,
    'fire': FireScenario,
    'earthquake': EarthquakeScenario,
    'tornado': TornadoScenario
}
