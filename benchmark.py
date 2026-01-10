import numpy as np
import matplotlib.pyplot as plt
from dyna_q_plus import StandardDynaQ, QLearning
from env.evacuation_env import make_evacuation_env
import config
import os

def train_agent(agent_class, name, episodes=500):
    env = make_evacuation_env(scenario='flood')
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    
    # Initialize agent
    if name == 'Dyna-Q':
        agent = agent_class(n_states, n_actions, planning_steps=50) # Use 50 planning steps
    else:
        agent = agent_class(n_states, n_actions) # Q-Learning has 0 planning steps
        
    survival_rates = []
    
    print(f"Training {name}...")
    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        
        # Track previous state for updates
        prev_positions = [env._pos_to_obs(p) for p in info['agent_positions']]
        prev_active = info['agent_active'].copy()
        
        while not done:
            actions = []
            for i, (pos, active) in enumerate(zip(info['agent_positions'], info['agent_active'])):
                if active:
                    agent_obs = env._pos_to_obs(pos)
                    action = agent.select_action(agent_obs)
                    actions.append(action)
                else:
                    actions.append(0)
            
            _, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            
            # --- CRITICAL FIX: PERFORM AGENT UPDATE ---
            num_agents = len(info.get('agent_active', [True]))
            for i, (pos, active) in enumerate(zip(info['agent_positions'], info['agent_active'])):
                if prev_active[i]: # Only update if agent was active in previous step
                    curr_obs = env._pos_to_obs(pos)
                    # Extract individual reward
                    agent_reward = info.get('agent_rewards', [0]*num_agents)[i]
                    # Update Q-table
                    agent.update(prev_positions[i], actions[i], agent_reward, curr_obs, not active or done)
            
            prev_positions = [env._pos_to_obs(p) for p in info['agent_positions']]
            prev_active = info['agent_active'].copy()
            
        # Calculate survival for this episode
        evacuated = sum(1 for i, active in enumerate(info.get('agent_active', []))
                       if not active and info['agent_positions'][i] in info['evacuation_zones'])
        survival = evacuated / config.NUM_AGENTS
        survival_rates.append(survival)
        
        agent.decay_epsilon() # Don't forget to decay epsilon!
        
        if (episode + 1) % 50 == 0:
            avg = np.mean(survival_rates[-50:])
            print(f"  Ep {episode+1}: {avg:.1%} survival | Epsilon: {agent.epsilon:.2f}")
            
    return survival_rates

# Run Benchmarks
# To get a cleaner graph, let's run slightly more episodes if possible, or smooth heavier
dyna_scores = train_agent(StandardDynaQ, 'Dyna-Q', episodes=5000)
q_scores = train_agent(QLearning, 'Q-Learning', episodes=5000)

# Smooth data for plotting
def smooth(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.figure(figsize=(12, 7))
plt.plot(smooth(dyna_scores), label='Dyna-Q (Model-Based)', linewidth=2.5, color='#1f77b4')
plt.plot(smooth(q_scores), label='Q-Learning (Model-Free)', linewidth=2.5, linestyle='--', color='#ff7f0e')
plt.title('Performance Comparison: Dyna-Q vs Q-Learning', fontsize=14)
plt.xlabel('Training Episodes', fontsize=12)
plt.ylabel('Survival Rate', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('Benchmark_Plot.png', dpi=300)
print("Plot saved to Benchmark_Plot.png")
