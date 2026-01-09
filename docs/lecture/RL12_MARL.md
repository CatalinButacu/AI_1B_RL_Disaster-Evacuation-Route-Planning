# RL12_MARL

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL12_MARL.pdf

**Pages:** 63

---


## Page 1

Reinforcement Learning 
12. Multi-Agent Reinforcement Learning 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

Multi-Agent Reinforcement Learning 
1. Multi-Agent Reinforcement Learning 
2. Independent Q-Learning (IQL) 
3. Centralized Training with Decentralized Execution (CTDE) 
4. Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
5. Multi-Agent Proximal Policy Optimization (MAPPO) 
6. MARL Approaches in Multi-Player Games 
2 


## Page 3

Multi-Agent Reinforcement Learning 
1. Multi-Agent Reinforcement Learning 
2. Independent Q-Learning (IQL) 
3. Centralized Training with Decentralized Execution (CTDE) 
4. Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
5. Multi-Agent Proximal Policy Optimization (MAPPO) 
6. MARL Approaches in Multi-Player Games 
3 


## Page 4

From Single-Agent RL to MARL 
Reinforcement Learning typically studies a single agent maximizing 
cumulative reward in an environment 
Many real-world systems involve multiple agents interacting in a 
shared environment 
Agents may interact competitively, cooperatively, or in mixed 
cooperative–competitive forms 
These interactions introduce complexities beyond the assumptions of 
single-agent RL algorithms 
Multi-Agent Reinforcement Learning (MARL) extends RL to handle 
such multi-agent environments 
4 


## Page 5

Complexity and State–Action Growth 
In multi-agent settings, each agent’s state and actions influence the 
global environment state 
Overall state space grows quickly as more agents and local state 
variables appear 
For n agents, each with m actions, joint action space size is mn 
Tabular methods like Q-learning become infeasible on exponentially 
large state–action spaces 
5 


## Page 6

Curse of Dimensionality 
Increased agent count produces very high-dimensional state and 
action representations 
Learning an optimal policy becomes computationally expensive in 
these large spaces 
Each agent must reason about how others’ actions affect its own 
future rewards 
Traditional single-agent algorithms assume more modest, tractable 
state spaces 
MARL seeks algorithms that tolerate and exploit large, coupled 
decision spaces 
6 


## Page 7

Dependencies and Non-Stationarity 
Single-agent RL often assumes environment dynamics depend only 
on one agent’s actions 
In MARL, dynamics depend on all agents’ actions and their changing 
policies 
When one agent improves its policy, others effectively experience a 
new environment 
This non-stationarity destabilizes learning for algorithms that 
assume fixed dynamics 
MARL algorithms address environments where transition and 
reward structures evolve during learning 
7 


## Page 8

Coordination and Communication 
Many MARL tasks require agents to coordinate actions to achieve 
shared goals 
Autonomous driving illustrates vehicles coordinating maneuvers to 
ensure safety and traffic efficiency 
Single-agent RL frameworks lack built-in mechanisms for multi-
agent coordination 
MARL introduces communication or information-sharing among 
agents when the task demands it 
Coordination mechanisms help agents reach better collective 
outcomes than independent learning 
8 


## Page 9

Defining Multi-Agent Reinforcement 
Learning 
MARL studies multiple agents learning simultaneously in a shared 
environment 
Each agent interacts with the environment and updates its behavior 
from experience 
One agent’s actions affect the environment state and the rewards of 
other agents 
Goal: each agent learns a policy that maximizes its long-term reward 
under multi-agent interaction 
Settings range from fully cooperative through purely competitive to 
mixed cooperative-competitive tasks 
9 


## Page 10

Motivation for MARL 
Many realistic domains involve several autonomous decision makers, 
not a single controller 
Single-agent RL cannot capture strategic interactions and mutual 
influence between agents 
Multi-agent environments appear non-stationary from each 
individual agent’s viewpoint 
MARL develops methods that adapt to other agents’ changing 
strategies 
This capability is essential for complex, dynamic tasks with 
interacting learners 
10 


## Page 11

Applications: Autonomous Driving and 
Robotics 
Autonomous driving uses MARL to manage interactions among many 
self-driving vehicles 
Vehicles must avoid collisions while optimizing traffic flow and travel 
efficiency 
MARL supports cooperative behaviors such as platooning and 
dynamic speed adjustment 
In robotics, multiple agents coordinate exploration, search-and-
rescue, or manufacturing tasks 
Swarm intelligence uses MARL to divide work, reduce redundancy, 
and handle uncertain environments 
11 


## Page 12

Applications: Games and Resource 
Management 
MARL powers agents in complex strategy games requiring both 
cooperation and competition 
Systems like AlphaStar learn to play StarCraft II at expert human 
levels 
Game agents manage resources, build units, and coordinate tactics 
across large environments 
In smart grids, MARL coordinates power generation, distribution, 
and consumption 
Resource-management agents balance local decisions with global 
efficiency and reliability 
12 


## Page 13

Fundamental MARL Concepts 
Each agent i has policy 𝜋𝑖𝑎𝑖
𝑠 specifying probabilities of actions 
𝑎𝑖 in state s 
Joint policy 𝝅= 𝜋1, … , 𝜋𝑛 describes behavior of all agents together 
Joint action space: 𝒜= 𝐴1 × ⋯× 𝐴𝑛 
Competitive rewards for agent i and shared cooperative reward: 
𝑅𝑖𝑠, 𝐚, 𝑠′  
Transition dynamics depend on joint actions: 𝑃𝑠′
𝑠, 𝐚  
13 


## Page 14

Multi-Agent Reinforcement Learning 
1. Multi-Agent Reinforcement Learning 
2. Independent Q-Learning (IQL) 
3. Centralized Training with Decentralized Execution (CTDE) 
4. Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
5. Multi-Agent Proximal Policy Optimization (MAPPO) 
6. MARL Approaches in Multi-Player Games 
14 


## Page 15

From Single-Agent to Multi-Agent Q-
Learning 
Standard Q-learning estimates 𝑄𝑠, 𝑎 as the expected return from 
action a in state s under an optimal policy 
The Q-learning update is: 
𝑄𝑠, 𝑎←𝑄𝑠, 𝑎+ 𝛼𝑟+ 𝛾max
𝑎′ 𝑄𝑠′, 𝑎′ −𝑄𝑠, 𝑎 
In multi-agent settings, multiple agents simultaneously select actions, 
producing joint transitions and coupled rewards 
Independent Q-Learning (IQL) treats each agent as isolated, applying 
Q-learning separately without coordination or modeling others 
IQL assumes other agents’ effects on state transitions and rewards 
are external, unmodeled environmental dynamics 
15 


## Page 16

Q-Function Updates in IQL 
Each agent iii maintains its own Q-function 𝑄𝑖𝑠, 𝑎𝑖, updated 
independently based on local actions and rewards 
The IQL update rule is: 
𝑄𝑖𝑠, 𝑎𝑖←𝑄𝑖𝑠, 𝑎𝑖+ 𝛼𝑟𝑖+ 𝛾max
𝑎𝑖
′ 𝑄𝑖𝑠′, 𝑎𝑖
′ −𝑄𝑖𝑠, 𝑎𝑖
 
The environment state s includes all agents’ positions and features, 
allowing the Q-function to condition on global state 
No agent explicitly observes or predicts other agents’ actions; their 
effects are absorbed into the observed transition 
IQL assumes transitions 𝑠, 𝑎𝑖, 𝑠′  result from marginalizing over the 
unknown joint action 𝑎𝑖, 𝑎−𝑖 
 
16 


## Page 17

Joint Action Space and Environmental 
Dynamics 
At each step, all agents act simultaneously, forming a joint action 
vector 𝐮= 𝑎1, 𝑎2, … , 𝑎𝑁 
The next state𝑠′ ∼𝑃𝑠′
𝑠, 𝐮 depends on the full joint action, not 
just individual agent actions 
Each agent receives a reward 𝑟𝑖= 𝑅𝑖𝑠, 𝐮 that reflects the effects of 
all agents’ behaviors 
IQL agents experience transitions from s to s' due to 𝑎𝑖​, but these 
transitions are non-stationary 
As each agent changes its policy over time, the effective environment 
for any agent becomes non-Markovian 
 
17 


## Page 18

Full Observability and State Encoding 
Fully observable states encode all agents’ positions, goals, and 
features in a structured vector or tensor 
For example, a grid-world state may be represented as a 3-channel 
tensor: one for each agent and one for goals 
The Q-function 𝑄𝑖𝑠, 𝑎𝑖 takes the global state s and the agent’s own 
action 𝑎𝑖​ as input 
When agent 2 is present, the state s contains features that alter agent 
1’s Q-values compared to being alone 
Despite no explicit modeling, the influence of other agents is 
embedded in the changing input state representations 
 
18 


## Page 19

Non-Stationarity and Learning Instability 
IQL environments are non-stationary due to simultaneous policy 
updates by all agents during learning 
This violates the Markov assumption required for Q-learning 
convergence in standard MDPs 
Each agent’s environment appears to change over time even if the 
physical environment is static 
In cooperative settings, agents may converge if their joint behaviors 
stabilize 
In competitive or mixed settings, instability can cause oscillation or 
divergence of learned policies 
 
19 


## Page 20

Implicit Modeling Through State 
Representation 
IQL agents do not construct predictive models of others’ actions or 
policies 
The full state s includes observable features such as positions, 
velocities, or goals of other agents 
Differences between states 𝑠1​ (agent alone) and 𝑠2​ (with others) lead 
to different Q-values for the same action 
This implicit modeling allows agents to learn reactive behaviors 
without estimating others’ intentions 
The environment design must ensure that all relevant agent 
information is embedded in the state input to the Q-function 
 
20 


## Page 21

Per-Agent Learning from Marginal 
Transitions 
IQL assumes transitions 𝑠, 𝑎𝑖, 𝑠′  are averaged over unknown joint 
action distributions 𝑃𝑎−𝑖 
The Q-function 𝑄𝑖𝑠, 𝑎𝑖= 𝔼 
𝛾𝑡𝑟𝑖
𝑡
∞
𝑡=0
depends on expectations 
over others’ behavior 
The dynamics seen by agent iii reflect marginal distributions, not a 
fixed transition model 
Each Q-update incorporates the effects of hidden, evolving agent 
policies into the observed outcome 
This approach works only if the other agents’ influence is consistent 
or stabilizes over time 
 
21 


## Page 22

Grid World Example with Structured State 
In a 5×5 grid world, the global state may be encoded as a 5×5×3 
tensor 
Channel 1 indicates Agent 1’s position; channel 2 shows Agent 2’s; 
channel 3 marks goal locations 
Each agent receives this full tensor as input to its Q-function 
𝑄𝑖𝑠, 𝑎𝑖, supporting reactive learning 
If Agent 2 is near Agent 1, the state input reflects proximity, 
influencing Agent 1’s action choices 
Coordination emerges implicitly, as agents learn which actions yield 
better outcomes in the presence of others 
 
22 


## Page 23

Advantages of Independent Q-Learning 
IQL is simple to implement using standard single-agent Q-learning 
algorithms per agent 
No explicit communication or coordination protocols are required 
between agents 
The method scales to many agents by avoiding centralized 
representations of joint actions 
Agents can operate in parallel, each using local Q-functions and 
global observations 
IQL often performs well in cooperative environments where shared 
goals reduce policy conflict 
 
23 


## Page 24

Limitations and Assumptions in IQL 
IQL breaks the Markov assumption due to changing transition 
dynamics caused by learning agents 
Q-learning guarantees no longer apply; convergence is not ensured 
in general 
In competitive or mixed settings, learning may become unstable or 
fail to converge 
The method assumes that either policies eventually stabilize or that 
the environment’s stochasticity absorbs the non-stationarity 
IQL is unsuitable for tasks requiring explicit reasoning about others’ 
intentions or tight inter-agent coordination 
 
24 


## Page 25

Multi-Agent Reinforcement Learning 
1. Multi-Agent Reinforcement Learning 
2. Independent Q-Learning (IQL) 
3. Centralized Training with Decentralized Execution (CTDE) 
4. Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
5. Multi-Agent Proximal Policy Optimization (MAPPO) 
6. MARL Approaches in Multi-Player Games 
25 


## Page 26

Centralized Training, Decentralized 
Execution 
CTDE allows agents to train using global state and joint action 
information but execute policies using only local observations 
During training, agents can access full environmental state s, actions 
𝑎1, … , 𝑎𝑁, and rewards 𝑟1, … , 𝑟𝑁​ 
During execution, each agent iii follows a decentralized policy 𝜋𝑖𝑜𝑖, 
using only local observation 𝑜𝑖 
This approach addresses non-stationarity caused by concurrent 
learning in multi-agent settings 
CTDE supports scalable, real-time execution while enabling 
coordinated learning through centralized feedback 
 
26 


## Page 27

Learning Together, Acting Alone 
CTDE reflects a natural learning strategy: full guidance during 
training, independent decision-making in deployment 
Like humans trained with teachers and peers, agents benefit from 
global context before acting solo in dynamic environments 
CTDE mirrors real systems, such as teams, swarms, and societies, 
that coordinate through shared learning, not constant 
communication 
The paradigm bridges practical constraints and theoretical 
challenges, creating robust multi-agent coordination frameworks 
It shows that shared training, even without runtime messaging, can 
create implicitly coordinated autonomous agents 
 
27 


## Page 28

Formal CTDE Framework and Components 
The environment is modeled as a Dec-POMDP 
𝑆, {𝐴𝑖}, 𝑃, {𝑟𝑖}, 𝑍, 𝑂, 𝛾 
The centralized critic uses the global state s and joint action 
𝐚= 𝑎1, … , 𝑎𝑁 to estimate value functions 
The critic is trained using temporal-difference loss: 
ℒ𝜃= 𝐸
𝑟+ 𝛾max
𝐚′ 𝑄tot 𝑠′, 𝐚′; 𝜃−−𝑄tot 𝑠, 𝐚; 𝜃
2  
Each agent learns an independent policy 𝜋𝑖𝑜𝑖, guided by the 
centralized value function or critic 
Centralized training exploits information unavailable at execution 
time, improving stability and coordination 
 
28 


## Page 29

Centralized Critic Methods 
Centralized critic methods separate actor and critic roles; actors are 
decentralized, critics use full state and joint actions 
In MADDPG, actors use local inputs 𝑜𝑖​, but critics are trained with s 
and a 
The actor gradient in MADDPG is ∇𝜓𝑖log 𝜋𝑖
𝑎𝑖
𝑜𝑖∇𝑎𝑖𝑄𝑖𝑠, 𝐚  
This framework allows decentralized actors to benefit from 
centralized critics during training only 
Centralized critics reduce instability by removing ambiguity from 
other agents’ policies during training 
 
29 


## Page 30

Value Function Factorization 
In cooperative tasks with shared rewards, the total value can be 
decomposed into agent-specific components 
QMIX factorizes 𝑄tot 𝑠, 𝐚 using a monotonic mixing network: 
𝑄tot 𝑠, 𝐚= 𝑓𝑄1 𝑜1, 𝑎1 , … , 𝑄𝑁𝑜𝑁, 𝑎𝑁; 𝑠 
The mixing network ensures that maximizing each 𝑄𝑖​ leads to the 
joint optimal action under constraints 
VDN, a simpler variant, uses additive decomposition 𝑄tot =  𝑄𝑖
𝑖
​, 
without state-dependent mixing 
These methods support decentralized execution while maintaining 
joint optimality during training 
 
30 


## Page 31

Strengths of CTDE 
Rich training signals from global state and actions enhance 
coordination and convergence stability 
CTDE mitigates non-stationarity caused by simultaneous multi-agent 
learning 
Policies trained under CTDE execute independently, avoiding 
communication overhead during deployment 
CTDE can align decentralized policies with a global objective, 
improving team performance 
Scalability is achieved by decoupling training complexity from 
execution runtime 
 
31 


## Page 32

Limitations and Challenges of CTDE 
Centralized training assumes availability of full environment state, 
which may not exist in real-world applications 
Joint action spaces grow exponentially with the number of agents, 
challenging critic scalability 
Mismatch between training (with full information) and execution 
(with partial observation) may reduce performance 
Designing effective critics or mixing networks is essential for stable 
and generalizable training 
CTDE may require simulation or full observability during training, 
limiting its use in partially observable or adversarial environments 
32 


## Page 33

Applications of CTDE 
CTDE methods are used in robotic teams, such as drones or 
warehouse robots trained with shared global state 
Self-driving vehicles use CTDE during simulation to learn 
cooperative driving behavior with access to joint information 
In games like StarCraft (SMAC benchmark), agents train via CTDE to 
coordinate strategies while executing independently 
Value factorization methods like QMIX and QPLEX have been applied 
to multi-robot path planning and cooperative control 
CTDE outperforms independent Q-learning by leveraging global 
training data while preserving decentralized policies 
 
33 


## Page 34

Comparison to Single-Agent Q-Learning 
Single-agent Q-learning assumes a stationary environment; 
transitions depend only on the agent's own actions 
In multi-agent settings, co-adapting policies violate this assumption 
and introduce non-stationarity 
CTDE mitigates these issues by incorporating other agents’ behavior 
into centralized training with global information 
During execution, decentralized policies operate robustly despite the 
dynamic environment 
Single-agent methods cannot adapt to the inter-agent dynamics that 
CTDE explicitly models during training 
 
34 


## Page 35

Concluding Remarks on CTDE 
CTDE enables agents to learn in rich, fully informed environments 
while acting autonomously after deployment 
Centralized critics and value decomposition networks support 
coordination without requiring communication at runtime 
CTDE remains a foundational design in modern MARL, balancing 
theory with real-world applicability 
Future research aims to improve critic scalability, mixing 
architectures, and robustness to training–execution mismatch 
CTDE enables complex behaviors in cooperative and mixed-motive 
environments that are infeasible under purely decentralized learning 
 
35 


## Page 36

Multi-Agent Reinforcement Learning 
1. Multi-Agent Reinforcement Learning 
2. Independent Q-Learning (IQL) 
3. Centralized Training with Decentralized Execution (CTDE) 
4. Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
5. Multi-Agent Proximal Policy Optimization (MAPPO) 
6. MARL Approaches in Multi-Player Games 
36 


## Page 37

DDPG Recap and Extension 
DDPG uses an actor-critic architecture for continuous action spaces, 
with off-policy learning and target networks 
The critic minimizes TD error using the target: 
  
𝑦= 𝑟+ 𝛾𝑄𝑠′, 𝜋𝑠′; 𝜃𝜋−; 𝜃𝑄− 
The critic loss is ℒ𝜃𝑄= 𝐸
𝑄𝑠, 𝑎; 𝜃𝑄−𝑦2 , with replay buffer 
sampling 
Extending DDPG to multi-agent settings requires addressing joint 
action effects and agent co-adaptation 
MADDPG introduces agent-specific centralized critics to handle the 
joint state-action space 
 
37 


## Page 38

Motivation for MADDPG 
Independent agents using DDPG suffer from unstable learning due to 
non-stationarity in multi-agent environments 
MADDPG extends DDPG to multi-agent systems using centralized 
training with decentralized execution (CTDE) 
Each agent’s critic is trained with access to global state and all agents’ 
actions, stabilizing the learning process 
Each actor relies only on local observations, enabling deployment 
without centralized information 
MADDPG supports coordination in continuous control tasks such as 
robotics and multi-agent navigation 
 
38 


## Page 39

Formal Framework and Setup 
MADDPG models the environment as a Dec-POMDP: 
𝑆, {𝐴𝑖}, 𝑃, {𝑟𝑖}, {𝑂𝑖}, 𝛾 
Each agent iii has an actor 𝜋𝑖𝑜𝑖; 𝜃𝑖
𝜋 using local observations 𝑜𝑖​ 
The centralized critic 𝑄𝑖𝑥, 𝐚; 𝜃𝑖
𝑄 is trained on the full state x and 
joint action a 
The critic evaluates how agent i’s action performs in the context of all 
agents’ actions 
This setup allows agents to learn policies aligned with global 
outcomes despite decentralized execution 
 
39 


## Page 40

Critic and Actor Updates 

For agent i, the critic’s TD target is 𝑦= 𝑟𝑖+ 𝛾𝑄𝑖𝑥′, 𝐚′; 𝜃𝑖
𝑄−, with 
𝑎𝑗
′ = 𝜋𝑗𝑜𝑗
′; 𝜃𝑗
𝜋− 

The critic loss is 
ℒ𝜃𝑖
𝑄= 𝐸
𝑄𝑖𝑥, 𝐚; 𝜃𝑖
𝑄−𝑦
2  

The actor gradient is 
∇𝜃𝑖
𝜋𝐽≈𝐸∇𝑎𝑖𝑄𝑖𝑥, 𝐚; 𝜃𝑖
𝑄∇𝜃𝑖
𝜋𝜋𝑖𝑜𝑖; 𝜃𝑖
𝜋
 

Updates use experiences from a shared replay buffer D, sampled off-policy 

Target networks for actor and critic are updated using soft updates with 
parameter τ 
 
40 


## Page 41

Core Features of MADDPG 
Centralized critics provide stronger learning signals by evaluating 
actions in global context 
Decentralized actors ensure scalability and deployment without 
shared observations 
Replay buffer and target networks help stabilize learning in dynamic 
environments 
The algorithm supports continuous control, making it well-suited for 
robotic and physical systems 
MADDPG accommodates cooperative and mixed-motive tasks 
without requiring communication during execution 
 
41 


## Page 42

Key Limitations and Open Issues 
The centralized critic must process large joint action and state 
spaces, increasing computational cost as agent count grows 
Replay buffer experiences may become stale if agents’ policies change 
significantly over time 
Execution under partial observability may diverge from training 
conditions that use full state 
Effective credit assignment remains difficult, especially when 
rewards are sparse or delayed 
Scalability and generalization depend on careful architecture design 
and training discipline 
 
42 


## Page 43

Comparison to Independent Methods 
Independent actor-critic methods lack stability in multi-agent 
environments due to unmodeled interactions 
MADDPG improves stability by incorporating joint state and actions 
into critic updates during training 
Shared global rewards can be leveraged more effectively with 
centralized critics than with independent learning 
Decentralized execution retains practical feasibility while improving 
learning through centralized feedback 
MADDPG achieves better coordination and efficiency than fully 
independent policy learning 
 
43 


## Page 44

Real-World Applications of MADDPG 
Autonomous vehicles train together for coordinated driving and then 
act independently using local sensors 
Multi-robot systems, such as delivery drones or manipulators, use 
MADDPG for collaborative control tasks 
MADDPG has shown strong performance in simulated multi-agent 
domains such as StarCraft Multi-Agent Challenge (SMAC) 
Continuous control problems in partially observable environments 
benefit from MADDPG’s centralized learning structure 
MADDPG supports robust, scalable policies without requiring real-
time inter-agent communication 
 
44 


## Page 45

Final Remarks on MADDPG 
MADDPG advances MARL by integrating centralized learning with 
decentralized, actor-only execution 
Centralized critics solve non-stationarity and support coordinated 
behavior in dynamic agent environments 
Replay-based, off-policy training with target networks adds stability 
to multi-agent learning 
The method is well suited for cooperative and continuous-action 
domains where global coordination is essential 
Continued research explores improvements in critic design, 
experience replay, and hybrid CTDE methods 
 
45 


## Page 46

Multi-Agent Reinforcement Learning 
1. Multi-Agent Reinforcement Learning 
2. Independent Q-Learning (IQL) 
3. Centralized Training with Decentralized Execution (CTDE) 
4. Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
5. Multi-Agent Proximal Policy Optimization (MAPPO) 
6. MARL Approaches in Multi-Player Games 
46 


## Page 47

PPO Foundations and Motivation 

PPO optimizes a clipped surrogate objective to ensure stable policy updates: 
𝐿PPO 𝜃= 𝐸𝑡min 𝑟𝑡𝜃𝐴𝑡
 , clip 𝑟𝑡𝜃, 1 −𝜖, 1 + 𝜖𝐴𝑡
 
 

The ratio 𝑟𝑡𝜃=
𝜋𝜃𝑎𝑡𝑠𝑡
𝜋𝜃old
𝑎𝑡𝑠𝑡 controls the update magnitude 

Clipping prevents drastic policy changes, improving learning stability and 
preventing policy collapse 

PPO performs well in single-agent environments with stationary dynamics 
and localized state inputs 

Multi-agent settings introduce non-stationarity, shared rewards, and 
coordination needs that PPO does not address 
 
47 


## Page 48

Challenges in Multi-Agent PPO 
Policy changes by one agent affect the environment observed by 
others, breaking stationarity assumptions 
Assigning credit for joint rewards across multiple agents complicates 
policy evaluation 
Coordinated behavior cannot emerge from independent agents 
optimizing in isolation 
Multi-agent learning dynamics require centralized information for 
stable training signals 
Direct application of PPO fails to handle joint interdependencies and 
temporal credit assignment 
 
48 


## Page 49

MAPPO and Centralized Training 
MAPPO extends PPO using Centralized Training with Decentralized 
Execution (CTDE) principles 
Each agent has a decentralized actor 𝜋𝜃𝑖𝑎𝑡
𝑖
𝑜𝑡
𝑖, dependent on 
local observations only 
A shared centralized critic evaluates joint action outcomes using 
global state and all agents’ actions 
This structure addresses non-stationarity by conditioning value 
estimates on full joint context 
During execution, agents act independently, ensuring practical 
deployment without communication 
 
49 


## Page 50

MAPPO Objective and Advantage 

Each agent’s PPO-style objective is: 
𝐿MAPPO 𝜃𝑖= 𝐸𝑡min 𝑟𝑡
𝑖𝜃𝑖𝐴𝑡
𝑖
 , clip 𝑟𝑡
𝑖𝜃𝑖, 1 −𝜖, 1 + 𝜖𝐴𝑡
𝑖
 
 

The probability ratio is: 
𝑟𝑡
𝑖𝜃𝑖=
𝜋𝜃𝑖𝑎𝑡
𝑖
𝑜𝑡
𝑖
𝜋𝜃𝑖,old 𝑎𝑡
𝑖
𝑜𝑡
𝑖 

The advantage estimate uses centralized value functions: 
𝐴𝑡
𝑖
 = 𝑄𝑠𝑡, 𝑎𝑡
1, … , 𝑎𝑡
𝑁−𝑉𝑠𝑡 

The critic evaluates how joint actions contribute to the reward, aiding credit 
assignment 

This advantage formulation promotes cooperation and improves learning signal 
quality 
 
50 


## Page 51

Key Differences from PPO 
Critic Structure: PPO uses local or global value functions; MAPPO 
requires a centralized critic with joint input 
Execution Mode: PPO acts through a single agent; MAPPO supports 
many agents acting independently 
Non-Stationarity Handling: PPO assumes fixed dynamics; MAPPO 
uses centralized critics to stabilize multi-agent training 
Coordination Capability: PPO optimizes isolated agents; MAPPO 
facilitates coordinated behavior through shared training signals 
Stability and Efficiency: Both inherit PPO’s clipped updates, but 
MAPPO enhances sample efficiency in cooperative environments 
 
51 


## Page 52

Advantages of MAPPO 
MAPPO maintains PPO’s policy stability via constrained updates and 
trust regions 
Centralized critics yield consistent advantage estimates that reduce 
variance and improve convergence 
Decentralized execution ensures real-world scalability and autonomy 
in runtime operation 
The algorithm supports cooperation by aligning agent updates 
through globally informed feedback 
MAPPO has demonstrated strong empirical performance in team-
based domains like multi-robot control 
 
52 


## Page 53

Remaining Challenges in MAPPO 
The centralized critic grows in complexity with more agents and 
larger joint action/state spaces 
The training-execution mismatch persists; actors may act 
suboptimally when deprived of training-time state access 
Credit assignment remains difficult in tasks with delayed or sparse 
rewards, even with joint critic feedback 
Learning stability depends on accurate advantage estimation, which 
hinges on critic design and global observability 
Scalability and efficiency tradeoffs emerge when balancing global 
input richness with computational tractability 
 
53 


## Page 54

Use Cases and Applications 
MAPPO performs well in robot soccer, where agents learn to pass, 
defend, and coordinate under shared objectives 
Autonomous fleets (e.g., drone swarms, vehicle platoons) benefit 
from centralized training for strategic coordination 
Multi-agent game environments like StarCraft use MAPPO to train 
units for collective combat and navigation 
The algorithm allows teams to learn coordinated policies without 
requiring runtime communication 
These applications highlight MAPPO’s blend of safe policy updates, 
coordination, and practical deployment 
 
54 


## Page 55

Summary and Final Observations 
MAPPO adapts PPO’s stable updates to cooperative multi-agent 
settings via centralized critics and decentralized actors 
It addresses core MARL problems: non-stationarity, credit 
assignment, and coordinated behavior 
During training, full-state access enables better policy evaluation and 
advantage estimation 
During execution, policies operate independently using only local 
observations 
MAPPO balances robustness and practicality, making it a key method 
for scalable multi-agent learning 
 
55 


## Page 56

Multi-Agent Reinforcement Learning 
1. Multi-Agent Reinforcement Learning 
2. Independent Q-Learning (IQL) 
3. Centralized Training with Decentralized Execution (CTDE) 
4. Multi-Agent Deep Deterministic Policy Gradient (MADDPG) 
5. Multi-Agent Proximal Policy Optimization (MAPPO) 
6. MARL Approaches in Multi-Player Games 
56 


## Page 57

MARL in Real-Time Strategy Games 
Real-time strategy games like StarCraft II and DotA 2 challenge 
multi-agent reinforcement learning systems 
Agents manage resources, plan strategies, and control units in 
dynamic, partially observable, real-time environments 
State and action spaces become enormous due to many units, 
abilities, maps, and game events 
Long time horizons require reasoning about early economic choices 
and later large-scale battles 
These properties motivate sophisticated deep RL and MARL 
algorithms for competitive and cooperative play 
57 


## Page 58

RTS Challenges for Reinforcement Learning 
Agents process high-dimensional observations, including raw pixels 
or structured features like unit positions and health 
Real-time control demands frequent action decisions while managing 
multiple units simultaneously 
Multi-agent interactions require coordination among units or heroes 
for effective attacks, defenses, and objectives 
Large action branching factors and long episodes challenge 
exploration and credit assignment 
Deep reinforcement learning approximates complex policies that 
map rich game states to suitable actions 
58 


## Page 59

Hierarchical Learning in RTS Games 
RTS decisions split naturally into macro strategy and micro-level unit 
control 
Macro level handles economy, base expansion, unit production, and 
long-term strategic planning 
Micro level manages precise movements, targeting, spell usage, and 
retreating during battles 
Hierarchical reinforcement learning separates these levels to simplify 
learning and improve coordination 
Systems like AlphaStar and OpenAI Five exploit macro–micro 
structure for effective play 
59 


## Page 60

Centralized Training and Decentralized 
Execution 
Centralized Training with Decentralized Execution (CTDE) addresses 
non-stationarity and coordination challenges 
During training, agents share information and learn using a 
centralized critic observing the global game state 
The critic evaluates joint actions and rewards, encouraging coherent 
multi-agent strategies 
After training, each agent acts independently based only on its local 
observations 
CTDE underpins both AlphaStar and OpenAI Five in their multi-
agent learning setups 
60 


## Page 61

AlphaStar for StarCraft II 
AlphaStar combines supervised learning from human replays with 
reinforcement learning for StarCraft II 
Initial imitation learning provides baseline strategies and tactics 
similar to expert human players 
League-based multi-agent training creates agents specializing in 
diverse strategies, such as aggressive or defensive play 
Self-play exposes agents to many strategies and counter-strategies 
across maps and situations 
AlphaStar uses a multi-agent variant of Proximal Policy Optimization 
(PPO) for stable policy updates 
61 


## Page 62

OpenAI Five and Population-Based Training 
OpenAI Five controls five DotA 2 heroes, each with unique abilities, 
against human teams 
Agents learn through massive self-play, improving coordination and 
tactics over thousands of games 
LSTM networks provide memory, enabling decisions under partial 
observability and long temporal dependencies 
Training uses CTDE with a centralized critic plus careful reward 
shaping and hyperparameter tuning 
Population-based training maintains diverse agents, promotes strong 
strategies, and improves robustness against varied opponents 
62 


## Page 63

Conclusions 

MARL extends single-agent RL to environments where multiple agents 
interact, cooperate, compete, and reshape each other’s dynamics 

Independent methods like IQL treat other agents as part of the 
environment, causing non-stationarity and limited coordination 

Centralized Training with Decentralized Execution stabilizes learning by 
training critics on global information while actors use local observations 

Actor-critic MARL algorithms such as MADDPG and MAPPO enable 
coordination in continuous or cooperative tasks with shared rewards 

Self-play, hierarchical control, and population-based training achieved 
superhuman performance in complex games like StarCraft II and DotA 2 
63 
