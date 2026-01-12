"""
Benchmark script for comparing RL algorithms across disaster scenarios.
Generates comparison plots for all 3 algorithms (Q-Learning, Dyna-Q, Dyna-Q+) per scenario.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from env.evacuation_env import make_evacuation_env
from algo import QLearning, DynaQ, DynaQPlus, AGENTS
import config

SCENARIOS = ['flood', 'fire', 'earthquake', 'tornado']
ALGORITHMS = ['random', 'q_learning', 'dyna_q', 'dyna_q_plus']
ALGORITHM_LABELS = {
    'random': 'Random Walk',
    'q_learning': 'Q-Learning',
    'dyna_q': 'Dyna-Q',
    'dyna_q_plus': 'Dyna-Q+'
}
ALGORITHM_COLORS = {
    'random': '#95A5A6',
    'q_learning': '#E74C3C',
    'dyna_q': '#3498DB',
    'dyna_q_plus': '#2ECC71'
}


def train_with_history(agent, env, num_episodes: int, verbose: bool = True):
    """Train agent and return episode-by-episode history."""
    episode_rewards, episode_survival = [], []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        total_reward, done = 0, False
        num_agents = len(info.get('agent_active', [True]))
        
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
            total_reward += reward
            
            for i, (pos, active) in enumerate(zip(info['agent_positions'], info['agent_active'])):
                if prev_active[i]:
                    curr_obs = env._pos_to_obs(pos)
                    agent_reward = info.get('agent_rewards', [0]*num_agents)[i]
                    agent.update(prev_positions[i], actions[i], agent_reward, curr_obs, not active or done)
            
            prev_positions = [env._pos_to_obs(p) for p in info['agent_positions']]
            prev_active = info['agent_active'].copy()
        
        evacuated = sum(1 for i, active in enumerate(info.get('agent_active', []))
                       if not active and info['agent_positions'][i] in info['evacuation_zones'])
        survival = evacuated / num_agents if num_agents > 0 else 0
        
        episode_rewards.append(total_reward)
        episode_survival.append(survival)
        agent.decay_epsilon(episode, num_episodes)
        
        if verbose and (episode + 1) % 500 == 0:
            avg_survival = np.mean(episode_survival[-500:])
            print(f"    Episode {episode + 1}/{num_episodes} | Survival: {avg_survival:.1%}")
    
    return episode_rewards, episode_survival


def evaluate_agent(agent, env, num_episodes: int = 100):
    """Evaluate trained agent and return rewards, survival, and arrival steps."""
    episode_rewards, episode_survival = [], []
    all_arrival_steps = []
    
    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward, done = 0, False
        num_agents = len(info.get('agent_active', [True]))
        steps = 0
        
        # Track which agents have arrived and at what step
        arrival_times = {} # agent_idx -> step
        prev_active = info['agent_active'].copy()
        
        while not done:
            steps += 1
            actions = []
            for i, (pos, active) in enumerate(zip(info['agent_positions'], info['agent_active'])):
                if active:
                    agent_obs = env._pos_to_obs(pos)
                    action = agent.select_action(agent_obs, greedy=True)
                    actions.append(action)
                else:
                    actions.append(0)
            
            _, reward, terminated, truncated, info = env.step(actions)
            done = terminated or truncated
            total_reward += reward
            
            # Record arrival time for newly inactive agents who are in exit zones
            for i, active in enumerate(info['agent_active']):
                if prev_active[i] and not active:
                    if info['agent_positions'][i] in info['evacuation_zones']:
                        arrival_times[i] = steps
            
            prev_active = info['agent_active'].copy()
            
        evacuated_indices = [i for i, active in enumerate(info.get('agent_active', []))
                           if not active and info['agent_positions'][i] in info['evacuation_zones']]
        survival = len(evacuated_indices) / num_agents if num_agents > 0 else 0
        
        episode_rewards.append(total_reward)
        episode_survival.append(survival)
        
        # Only collect steps for agents that actually reached the exit
        for i in evacuated_indices:
            if i in arrival_times:
                all_arrival_steps.append(arrival_times[i])
    
    return np.mean(episode_rewards), np.mean(episode_survival), all_arrival_steps


def smooth_curve(data, window=100):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')


def run_benchmark(scenarios=None, episodes=3000):
    """Run full benchmark across scenarios and algorithms."""
    scenarios = scenarios or SCENARIOS
    results = {}
    
    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {scenario.upper()}")
        print(f"{'='*70}")
        
        results[scenario] = {}
        
        for algo_name in ALGORITHMS:
            print(f"\n  Training {ALGORITHM_LABELS[algo_name]}...")
            
            env = make_evacuation_env(scenario=scenario)
            agent_class = AGENTS[algo_name]
            agent = agent_class(env.observation_space.n, env.action_space.n)
            
            rewards, survival = train_with_history(agent, env, episodes, verbose=True)
            
            results[scenario][algo_name] = {
                'rewards': rewards,
                'survival': survival,
                'final_survival': np.mean(survival[-100:]) if len(survival) >= 100 else np.mean(survival)
            }
            
            # Save trained model
            model_dir = config.get_models_dir(algo_name)
            os.makedirs(model_dir, exist_ok=True)
            agent.save(f"{model_dir}/{algo_name}_{scenario}.npz")
            print(f"    Final survival: {results[scenario][algo_name]['final_survival']:.1%}")
    
    return results


def run_evaluation(scenarios=None, test_episodes=100):
    """Evaluate existing trained models across scenarios and algorithms."""
    scenarios = scenarios or SCENARIOS
    results = {}
    
    print(f"\n{'='*95}")
    print(f"{'EVALUATION MODE: USING EXISTING MODELS':^95}")
    print(f"{'='*95}")
    print(f"{'Algorithm':<15} | {'Surv %':<8} | {'Avg Steps':<10} | {'Min Steps':<10} | {'Max Steps':<10}")
    print(f"{'-'*95}")
    
    for scenario in scenarios:
        results[scenario] = {}
        print(f"\nScenario: {scenario.upper()}")
        
        for algo_name in ALGORITHMS:
            env = make_evacuation_env(scenario=scenario)
            agent_class = AGENTS[algo_name]
            agent = agent_class(env.observation_space.n, env.action_space.n)
            
            model_path = f"{config.get_models_dir(algo_name)}/{algo_name}_{scenario}.npz"
            if os.path.exists(model_path):
                try:
                    agent.load(model_path)
                    avg_reward, avg_survival, arrival_steps = evaluate_agent(agent, env, test_episodes)
                    
                    mean_steps = np.mean(arrival_steps) if arrival_steps else 0
                    min_steps = np.min(arrival_steps) if arrival_steps else 0
                    max_steps = np.max(arrival_steps) if arrival_steps else 0
                    
                    results[scenario][algo_name] = {
                        'rewards': [avg_reward], 
                        'survival': [avg_survival],
                        'final_survival': avg_survival,
                        'steps_mean': mean_steps,
                        'steps_min': min_steps,
                        'steps_max': max_steps
                    }
                    print(f"  {ALGORITHM_LABELS[algo_name]:<15} | {avg_survival:^8.1%} | {mean_steps:^10.1f} | {min_steps:^10} | {max_steps:^10}")
                except Exception as e:
                    print(f"  {ALGORITHM_LABELS[algo_name]:<15} | Failed to load: {e}")
            else:
                print(f"  {ALGORITHM_LABELS[algo_name]:<15} | Missing model: {model_path}")
                
    return results


def plot_scenario_comparison(results, scenario, save_dir="outputs/plots"):
    """Generate comparison plot for a single scenario."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{scenario.upper()} Scenario - Algorithm Comparison', fontsize=16, fontweight='bold')
    
    for algo_name in ALGORITHMS:
        if algo_name not in results[scenario]:
            continue
        
        data = results[scenario][algo_name]
        survival = smooth_curve(data['survival'])
        rewards = smooth_curve(data['rewards'])
        
        color = ALGORITHM_COLORS[algo_name]
        label = f"{ALGORITHM_LABELS[algo_name]} (Final: {data['final_survival']:.1%})"
        
        ax1.plot(np.array(survival) * 100, label=label, color=color, linewidth=2)
        ax2.plot(rewards, label=ALGORITHM_LABELS[algo_name], color=color, linewidth=2)
    
    ax1.set_title('Survival Rate Over Training', fontsize=12)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Survival Rate (%)')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    ax2.set_title('Episode Rewards Over Training', fontsize=12)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Reward')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = f"{save_dir}/benchmark_{scenario}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def plot_all_scenarios_summary(results, save_dir="outputs/plots"):
    """Generate 2x2 grid summary plot for all scenarios."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Algorithm Performance Across All Disaster Scenarios', fontsize=16, fontweight='bold')
    
    for idx, scenario in enumerate(SCENARIOS):
        if scenario not in results:
            continue
        
        ax = axes[idx // 2, idx % 2]
        
        for algo_name in ALGORITHMS:
            if algo_name not in results[scenario]:
                continue
            
            data = results[scenario][algo_name]
            survival = smooth_curve(data['survival'])
            color = ALGORITHM_COLORS[algo_name]
            label = f"{ALGORITHM_LABELS[algo_name]} ({data['final_survival']:.0%})"
            
            ax.plot(np.array(survival) * 100, label=label, color=color, linewidth=2)
        
        ax.set_title(f'{scenario.upper()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Survival Rate (%)')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    
    plot_path = f"{save_dir}/benchmark_all_scenarios.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {plot_path}")
    plt.close()


def generate_results_table(results):
    """Print final results summary table."""
    headers = ["Scenario"] + [ALGORITHM_LABELS[a] for a in ALGORITHMS]
    header_str = " | ".join([f"{h:<15}" for h in headers])
    
    print("\n" + "=" * len(header_str))
    print("FINAL RESULTS SUMMARY")
    print("=" * len(header_str))
    print(header_str)
    print("-" * len(header_str))
    
    for scenario in SCENARIOS:
        if scenario not in results:
            continue
        
        row = f"{scenario.upper():<15}"
        for algo in ALGORITHMS:
            if algo in results[scenario]:
                surv = results[scenario][algo]['final_survival']
                row += f" | {surv:>15.1%}"
            else:
                row += f" | {'N/A':>15}"
        print(row)
    
    print("=" * len(header_str))


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark or Evaluate algorithms')
    parser.add_argument('--mode', type=str, default='eval', choices=['train', 'eval'], 
                        help='train: Retrain from scratch | eval: Use existing models')
    parser.add_argument('--episodes', type=int, default=3000, help='Episodes per algorithm (train/eval)')
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS, help='Scenarios to run')
    args = parser.parse_args()
    
    print("="*70)
    print(f"DISASTER EVACUATION {args.mode.upper()}")
    print(f"Episodes: {args.episodes} | Scenarios: {args.scenarios}")
    print(f"Grid: {config.GRID_SIZE}x{config.GRID_SIZE} | Agents: {config.NUM_AGENTS}")
    print("="*70)
    
    if args.mode == 'train':
        # Run full training benchmark
        results = run_benchmark(scenarios=args.scenarios, episodes=args.episodes)
        
        # Generate per-scenario comparison plots
        for scenario in args.scenarios:
            if scenario in results:
                plot_scenario_comparison(results, scenario)
        
        # Generate summary plot if all scenarios were run
        if len(args.scenarios) == len(SCENARIOS):
            plot_all_scenarios_summary(results)
    else:
        # Run evaluation of existing models
        results = run_evaluation(scenarios=args.scenarios, test_episodes=args.episodes if args.episodes < 1000 else 100)
    
    # Print results table
    generate_results_table(results)
    
    print("\nProcess complete!")


if __name__ == "__main__":
    main()
