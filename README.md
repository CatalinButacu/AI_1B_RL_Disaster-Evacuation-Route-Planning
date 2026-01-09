# Project #7: Disaster Evacuation Route Planning

**Finding optimal evacuation routes in uncertain environments using Dyna-Q**

## Project Description
> "Find optimal evacuation routes in an uncertain environment."
> Algorithm: **Dyna-Q** | Environment: **custom env**

## Overview

Multi-agent reinforcement learning for disaster evacuation with:
- Dynamic hazards (flood, fire, earthquake, tornado)
- Multiple agents finding optimal routes to exits
- Obstacles and uncertain environment
- **Dyna-Q model-based planning** (learned environment model + Q-learning)

## Quick Start

```bash
# Train 5k episodes (achieves ~99% survival)
python main.py --train --scenario flood --episodes 5000

# Live demo
python main.py --demo --scenario flood --live

# Train all scenarios
python main.py --train --scenario all --episodes 5000

# Save demo as GIF
python main.py --demo --scenario flood
```

## Algorithm: Dyna-Q

Dyna-Q combines:
1. **Direct RL**: Q-learning on real experience
2. **Model learning**: Store (s, a) → (r, s') transitions
3. **Planning**: Simulate n steps using learned model

```
For each real step:
  1. Take action, get reward, next state
  2. Q-learning update: Q(s,a) += α[r + γ max Q(s') - Q(s,a)]
  3. Update model: Model(s,a) ← (r, s')
  4. Planning: repeat n times:
     - Sample random (s,a) from model
     - Get (r,s') from model
     - Q-learning update
```

## Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| GRID_SIZE | 12 | Map size (12×12) |
| NUM_AGENTS | 5 | People to evacuate |
| NUM_EXITS | 1 | Evacuation points |
| NUM_OBSTACLES | 10 | Blocked cells |
| PLANNING_STEPS | 50 | Dyna-Q planning iterations |

## Scenarios (Uncertain Environment)

| Scenario | Uncertainty Pattern |
|----------|---------------------|
| flood | Water spreads from corner (deterministic but unknown timing) |
| fire | Circular spread from center (stochastic spread rate) |
| earthquake | Random building collapse (highly stochastic) |
| tornado | Moving hazard zone (unpredictable path) |

## Project Structure

```
project/
    main.py             # Training & visualization
    config.py           # Configuration
    dyna_q_plus.py      # RL agents (Dyna-Q, Q-learning, SARSA, etc.)
    env/
        evacuation_env.py    # Custom gymnasium environment
        hazard_dynamics.py   # Hazard scenarios
    outputs/
        models/         # Trained agents (.npz)
        plots/          # Visualizations (.gif)
```

## Results

With Dyna-Q and individual agent rewards, achieves **99% survival rate** on flood scenario after 5000 episodes of training.
