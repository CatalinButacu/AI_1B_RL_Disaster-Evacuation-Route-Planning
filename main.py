import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import argparse
import os

from env.evacuation_env import EvacuationEnv, make_evacuation_env
from algo import QLearning, DynaQ, AGENTS
import config

SCENARIOS = ['flood', 'fire', 'earthquake', 'tornado']

COLORS = {
    'ground': '#7CBA5F',
    'road': '#A9A9A9',
    'flood': '#3498DB',
    'fire': '#E74C3C',
    'earthquake': '#8B4513',
    'tornado': '#5D3FD3',
    'evacuation': '#2ECC71',
    'building': '#BDC3C7',
    'agent_body': '#F5CBA7',
    'agent_shirt': ['#E74C3C', '#3498DB', '#9B59B6', '#1ABC9C', '#F39C12', '#E91E63', '#00BCD4', '#8BC34A', '#FF5722', '#673AB7'],
    'obstacle': '#7F8C8D',
    'safe_zone': '#27AE60'
}


def train(agent, env: EvacuationEnv, num_episodes: int, verbose: bool = True):
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
        
        if verbose and (episode + 1) % config.EVAL_FREQUENCY == 0:
            avg_reward = np.mean(episode_rewards[-config.EVAL_FREQUENCY:])
            avg_survival = np.mean(episode_survival[-config.EVAL_FREQUENCY:])
            print(f"Episode {episode + 1}/{num_episodes} | Reward: {avg_reward:.1f} | Survival: {avg_survival:.1%} | ε: {agent.epsilon:.3f}")
    
    return episode_rewards, episode_survival

def train_scenario(scenario: str, agent_name: str, episodes: int = 1000, resume: bool = False):
    print(f"\n{'='*60}")
    print(f"TRAINING: {scenario.upper()} | Agent: {agent_name} | Grid: {config.GRID_SIZE}x{config.GRID_SIZE}")
    print(f"{'='*60}")
    
    env = make_evacuation_env(scenario=scenario)
    agent_class = AGENTS[agent_name]
    agent = agent_class(env.observation_space.n, env.action_space.n)
    
    model_dir = config.get_models_dir(agent_name)
    model_path = f"{model_dir}/{agent_name}_{scenario}.npz"
    if resume and os.path.exists(model_path):
        agent.load(model_path)
        print(f"Resumed from: {model_path}")
    
    rewards, survival = train(agent, env, episodes)
    
    os.makedirs(model_dir, exist_ok=True)
    agent.save(model_path)
    
    final_survival = np.mean(survival[-100:]) if len(survival) >= 100 else np.mean(survival)
    print(f"Final Survival: {final_survival:.1%} | Saved: {model_path}")
    
    return agent, final_survival

class EvacuationDemo:
    def __init__(self, env: EvacuationEnv, agent, scenario: str):
        self.env = env
        self.agent = agent
        self.scenario = scenario
        self.fig, self.ax = plt.subplots(figsize=(16, 16))
        self.grid_patches = {}
        self.grid_texts = {}
        self.agent_artists = []
        self.info_text = None
        self.legend_text = None
        self.total_reward = 0
        self.step_count = 0
        self.done = False

    def draw_person(self, x, y, color_idx, active=True):
        artists = []
        shirt_color = COLORS['agent_shirt'][color_idx % len(COLORS['agent_shirt'])]
        
        if not active:
            return artists
        
        head = patches.Circle((x, y + 0.25), 0.12, facecolor=COLORS['agent_body'], 
                              edgecolor='#2C3E50', linewidth=1.5, zorder=15)
        artists.append(head)
        
        body = patches.FancyBboxPatch((x - 0.1, y - 0.15), 0.2, 0.35,
                                      boxstyle="round,pad=0.02,rounding_size=0.05",
                                      facecolor=shirt_color, edgecolor='#2C3E50', 
                                      linewidth=1, zorder=14)
        artists.append(body)
        
        left_leg = patches.Rectangle((x - 0.08, y - 0.35), 0.06, 0.22, 
                                     facecolor='#2C3E50', zorder=13)
        right_leg = patches.Rectangle((x + 0.02, y - 0.35), 0.06, 0.22, 
                                      facecolor='#2C3E50', zorder=13)
        artists.extend([left_leg, right_leg])
        
        for artist in artists:
            self.ax.add_patch(artist)
        
        return artists

    def setup(self):
        gs = self.env.grid_size
        self.ax.set_xlim(-1, gs + 1)
        self.ax.set_ylim(-2, gs + 1)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.fig.patch.set_facecolor('#1C2833')
        
        title = f"DISASTER EVACUATION: {self.scenario.upper()}"
        subtitle = f"{gs}x{gs} Grid | {len(self.env.agent_positions)} People"
        self.ax.set_title(f"{title}\n{subtitle}", fontsize=16, fontweight='bold', 
                         color='white', pad=15, family='sans-serif')
        
        for row in range(gs):
            for col in range(gs):
                y = gs - row - 1
                
                rect = patches.FancyBboxPatch(
                    (col + 0.02, y + 0.02), 0.96, 0.96,
                    boxstyle="round,pad=0.01,rounding_size=0.05",
                    facecolor=COLORS['ground'], edgecolor='#5D6D7E', linewidth=0.5
                )
                self.ax.add_patch(rect)
                self.grid_patches[(row, col)] = rect
                
                text = self.ax.text(col + 0.5, y + 0.5, '', ha='center', va='center',
                                   fontsize=7, color='white', fontweight='bold', zorder=5)
                self.grid_texts[(row, col)] = text
        
        for i in range(len(self.env.agent_positions)):
            self.agent_artists.append([])
        
        self.info_text = self.ax.text(
            0.5, -0.08, '', transform=self.ax.transAxes, fontsize=12, 
            ha='center', color='white', fontweight='bold',
            bbox={'boxstyle': 'round,pad=0.5', 'facecolor': '#2C3E50', 
                  'edgecolor': '#3498DB', 'alpha': 0.95}
        )
        
        legend_items = ['[G] Safe Zone', '[B] Hazard', '[#] Obstacle', '[*] Person']
        self.legend_text = self.ax.text(
            0.02, 0.02, '  '.join(legend_items), transform=self.ax.transAxes,
            fontsize=10, color='white', alpha=0.8
        )

    def update(self):
        hazard_color = COLORS.get(self.scenario, COLORS['flood'])
        gs = self.env.grid_size
        
        for row in range(gs):
            for col in range(gs):
                pos = (row, col)
                patch = self.grid_patches[pos]
                text = self.grid_texts[pos]
                
                if pos in self.env.evacuation_zones:
                    patch.set_facecolor(COLORS['safe_zone'])
                    patch.set_edgecolor('#1D8348')
                    patch.set_linewidth(2)
                    text.set_text('EXIT')
                    text.set_color('#004d00')
                elif pos in self.env.obstacles:
                    patch.set_facecolor(COLORS['obstacle'])
                    patch.set_edgecolor('#566573')
                    text.set_text('#')
                    text.set_color('#333')
                elif self.env.hazard.is_hazard(pos):
                    patch.set_facecolor(hazard_color)
                    patch.set_edgecolor(hazard_color)
                    patch.set_alpha(0.8)
                    char = '🌀' if self.scenario == 'tornado' else '~~'
                    text.set_text(char)
                    text.set_color('#FFFFFF')
                else:
                    patch.set_facecolor(COLORS['ground'])
                    patch.set_edgecolor('#5D6D7E')
                    patch.set_linewidth(0.5)
                    patch.set_alpha(1.0)
                    text.set_text('')
        
        for artists in self.agent_artists:
            for a in artists:
                a.remove()
        self.agent_artists = [[] for _ in range(len(self.env.agent_positions))]
        
        evacuated = 0
        dead = 0
        
        for i, (pos, active) in enumerate(zip(self.env.agent_positions, self.env.agent_active)):
            if active:
                x = pos[1] + 0.5
                y = self.env.grid_size - pos[0] - 0.5
                self.agent_artists[i] = self.draw_person(x, y, i, active=True)
            else:
                if pos in self.env.evacuation_zones:
                    evacuated += 1
                else:
                    dead += 1
        
        hazards = len(self.env.hazard.get_hazard_positions())
        active = sum(self.env.agent_active)
        total = len(self.env.agent_positions)
        
        status = f"Step {self.step_count}/{config.MAX_STEPS_PER_EPISODE}"
        status += f" | Active: {active}"
        status += f" | Evacuated: {evacuated}"
        status += f" | Dead: {dead}"
        status += f" | Hazards: {hazards}"
        self.info_text.set_text(status)

    def step(self):
        if self.done:
            return
        
        actions = []
        for i, (pos, active) in enumerate(zip(self.env.agent_positions, self.env.agent_active)):
            if active:
                agent_obs = self.env._pos_to_obs(pos)
                action = self.agent.select_action(agent_obs, greedy=True)
                actions.append(action)
            else:
                actions.append(0)
        
        _, reward, terminated, truncated, _ = self.env.step(actions)
        self.total_reward += reward
        self.step_count += 1
        self.done = terminated or truncated
        self.update()

    def animate(self, _):
        self.step()
        artists = list(self.grid_patches.values())
        for agent_arts in self.agent_artists:
            artists.extend(agent_arts)
        artists.append(self.info_text)
        return artists

    def run(self, save_path: str = None):
        self.env.reset()
        self.total_reward, self.step_count, self.done = 0, 0, False
        self.setup()
        self.update()
        
        anim = FuncAnimation(self.fig, self.animate, frames=config.MAX_STEPS_PER_EPISODE,
                            interval=150, blit=False, repeat=False)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            anim.save(save_path, writer='pillow', fps=6)
            print(f"Saved: {save_path}")
        else:
            plt.show()
        plt.close()

def demo(scenario: str, agent_name: str, live: bool = True):
    env = make_evacuation_env(scenario=scenario)
    agent_class = AGENTS[agent_name]
    agent = agent_class(env.observation_space.n, env.action_space.n)
    
    model_dir = config.get_models_dir(agent_name)
    model_path = f"{model_dir}/{agent_name}_{scenario}.npz"
    if os.path.exists(model_path):
        agent.load(model_path)
        print(f"Loaded: {model_path}")
    else:
        print("No model found. Training first...")
        train(agent, env, 500)
        os.makedirs(model_dir, exist_ok=True)
        agent.save(model_path)
    
    viz = EvacuationDemo(env, agent, scenario)
    if live:
        viz.run()
    else:
        plot_dir = config.get_plots_dir(agent_name)
        os.makedirs(plot_dir, exist_ok=True)
        viz.run(f"{plot_dir}/evacuation_{scenario}_{agent_name}.gif")

def main():
    parser = argparse.ArgumentParser(description='Disaster Evacuation Route Planning')
    parser.add_argument('--scenario', type=str, default='all', choices=SCENARIOS + ['all'])
    parser.add_argument('--agent', type=str, default='all', choices=list(AGENTS.keys()) + ['all'])
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--continue', dest='resume', action='store_true', help='Resume from saved model')
    parser.add_argument('--demo', action='store_true', help='Run visualization demo')
    parser.add_argument('--live', action='store_true', help='Show live window instead of saving GIF')
    parser.add_argument('--episodes', type=int, default=config.NUM_EPISODES, help='Training episodes')
    args = parser.parse_args()
    
    # If no specific mode (train/demo) is requested, default to both
    do_train = args.train
    do_demo = args.demo
    if not do_train and not do_demo:
        do_train = True
        do_demo = True

    scenarios = SCENARIOS if args.scenario == 'all' else [args.scenario]
    agents = list(AGENTS.keys()) if args.agent == 'all' else [args.agent]
    
    if do_train:
        print(f"\nStarting training for {len(agents)} agents across {len(scenarios)} scenarios...")
        for a_name in agents:
            results = {}
            for s in scenarios:
                _, survival = train_scenario(s, a_name, args.episodes, resume=args.resume)
                results[s] = survival
            
            if len(scenarios) > 1:
                print(f"\nTraining Results for agent: {a_name}")
                for s, surv in results.items():
                    print(f"  {s:12s}: {surv:.1%} survival")
    
    if do_demo:
        print(f"\nStarting demonstrations for {len(agents)} agents across {len(scenarios)} scenarios...")
        for a_name in agents:
            for s in scenarios:
                demo(s, a_name, live=args.live)

if __name__ == "__main__":
    main()
