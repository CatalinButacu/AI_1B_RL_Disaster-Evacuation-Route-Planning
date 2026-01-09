# RL13_Advanced

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL13_Advanced.pdf

**Pages:** 43

---


## Page 1

Reinforcement Learning 
13. Advanced Topics in Reinforcement Learning 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

Advanced Topics in RL 
1. Imitation Learning, RL from Human Feedback  
2. Offline RL 
3. Safe RL 
4. Hierarchical RL 
 
 
2 


## Page 3

Imitation learning and RLHF: overall goal 
Both methods steer agents toward human-like, decent behavior 
Imitation learning copies expert actions in observed states 
RLHF learns what humans like and optimizes for that objective 
Imitation focuses on actions; RLHF focuses on outcomes and 
preferences 
Both rely on the same underlying policy machinery 
3 


## Page 4

Behavior cloning: basic imitation setup 
There is an expert policy acting in the environment, possibly human 
or learned 
We record demonstration data as (s, a) pairs from expert trajectories 
States s describe situations; actions a are expert choices 
Goal: learn policy 𝜋𝜃𝑎
𝑠 that matches expert behavior 
Central question: “In these expert-like states, what would the expert 
do?” 
4 


## Page 5

Behavior cloning as supervised learning 
Treat (s, a) pairs exactly like labeled supervised examples 
Learn 𝜋𝜃𝑎
𝑠 with standard supervised training: 
𝜋𝜃𝑎
𝑠
≈expert’s action distribution 
For discrete actions, use cross-entropy loss with softmax over actions 
For continuous actions, use mean squared error on action vectors 
Network views states as inputs and expert actions as labels 
5 


## Page 6

Behavior cloning: lack of rollout awareness 
Supervised loss ignores downstream consequences of actions 
The model does not ask what happens several steps after an action 
A locally correct action may still lead to future failure 
Horizon length matters; one-step errors can cascade over time 
This blind spot motivates analysis of covariate shift 
6 


## Page 7

Covariate shift and distribution mismatch 
Training states come from expert trajectories near a “good manifold” 
in state space 
Deployed policy is imperfect; small action errors occur 
These errors move the system into states absent from the 
demonstrations 
Policy predictions degrade further on these unfamiliar states 
This mismatch in state distributions is covariate shift 
7 


## Page 8

Compounding errors and stability issues 
Small initial imitation errors appear at early time steps 
Subsequent states drift further from expert trajectories 
The agent eventually reaches highly unfamiliar, poorly modeled 
regions 
Performance can collapse over long horizons despite low per-step 
error 
Robotics often experiences drift and awkward poses from this 
phenomenon 
8 


## Page 9

Fixes for distribution drift (high-level) 
Dataset aggregation brings in states visited by the learned policy, for 
example DAgger 
Regularization keeps the learned policy near behavior in the 
demonstration dataset 
Control-theoretic methods impose explicit stability conditions on the 
closed-loop system 
Despite fixes, core limitation remains: supervised imitation ignores 
feedback from its own errors 
Future state distribution depends on the learned policy, not the 
expert 
9 


## Page 10

RLHF motivation: beyond low-level action 
labels 
Humans may handle tasks with complex long-term trade-offs poorly 
at the action level 
We care about fuzzy goals: helpfulness, politeness, safety, avoidance 
of harm 
Many environments lack clean, trustworthy reward signals 
Handcrafted rewards often produce bad incentives or unintended 
behaviors 
RLHF instead asks humans which trajectories or outputs they prefer 
10 


## Page 11

Preference data and reward modeling 
Collect comparisons: context x with two outputs 𝑦better , 𝑦worse 
Each data point: humans prefer 𝑦better over 𝑦worse for context x 
Train reward model 𝑟𝜙𝑥, 𝑦 using a Bradley-Terry style likelihood: 
𝑃prefer 𝑦better = 𝜎𝑟𝜙𝑥, 𝑦better −𝑟𝜙𝑥, 𝑦worse
 
Architecture usually reuses policy backbone with a scalar reward 
head 
Resulting reward model predicts how much humans like each output 
11 


## Page 12

RL on learned reward and relationship to 
imitation 
RL phase: sample trajectories, score with 𝑟𝜙, update policy via PPO 
PPO objective often includes a KL penalty to stay near a base 
supervised policy 
KL penalty limits reward-model exploitation and preserves safe, 
useful behavior 
Practical pipeline: pretraining, supervised imitation, then RLHF fine-
tuning 
Imitation copies expert behavior in-distribution; RLHF optimizes 
human-shaped objectives under distribution shift 
12 


## Page 13

Offline reinforcement learning: core idea 
Offline RL learns a policy from a fixed dataset of past experience only 
The algorithm receives the dataset upfront and collects no additional 
transitions 
No exploration, no “try it and see”; interaction with environment is 
forbidden 
Goal: extract a high-performing policy under this strict no-new-
samples regime 
Attractive for domains where online experimentation is expensive or 
unsafe 
13 


## Page 14

Offline vs standard off-policy RL 
Standard off-policy RL also learns from replayed past experience 
However, it still interacts with the environment while training 
The agent eventually tries poorly supported actions and observes bad 
returns 
These new samples correct over-optimistic Q-values for risky actions 
Offline RL removes this safety net; the replay buffer becomes the 
entire world 
14 


## Page 15

Behavior policy, learned policy, and 
distribution shift 
Dataset arises from a behavior policy 𝛽𝑎
𝑠, often unknown 
Offline RL seeks a new policy 𝜋𝑎
𝑠 that outperforms 𝛽 
State-action pairs in data follow 𝛽; 𝜋 may choose very different 
actions 
Result: distribution shift between visitation distributions of 𝜋 and 𝛽  
Dataset typically covers a small subset of all possible state-action 
pairs 
15 


## Page 16

Out-of-distribution actions and 
extrapolation 
Actions outside dataset support are out-of-distribution (OOD) 
relative to the logged data 
Online RL explores OOD actions and learns their consequences from 
experience 
Offline RL must estimate values for OOD actions without observing 
real outcomes 
Function approximators extrapolate Q-values from nearby in-
distribution samples 
Extrapolation errors in these unsupported regions create major 
offline RL difficulties 
16 


## Page 17

Naïve Q-learning and the deadly triad 
Standard Q-learning update uses a bootstrapped target:  
target = 𝑟+ 𝛾max
𝑎′ 𝑄𝑠′, 𝑎′  
Bootstrapping updates Q(s, a) using Q-values at the next state s' 
Off-policy learning with function approximation already risks 
instability or divergence 
This combination is the “deadly triad”: function approximation, 
bootstrapping, off-policy updates 
Offline RL intensifies these issues because the dataset remains fixed 
17 


## Page 18

Feedback loop from hallucinated high  
Q-values 
Dataset contains only actions actually taken; backups still maximize 
over all possible actions 
Function approximation can assign huge Q-values to unseen actions 
at state s' 
No datapoint contradicts these fantasy values, so they persist 
The max operator repeatedly selects these inflated values in targets 
Over time, optimism spreads through Q-values and yields disastrous 
greedy policies 
18 


## Page 19

Pessimism principle in offline RL 
Modern offline RL adopts deliberate pessimism about poorly 
supported actions 
If data coverage seems weak, algorithms underestimate that action’s 
value 
Pessimism discourages the policy from assigning probability to OOD 
actions 
Policies remain near regions where Q-values reflect actual experience 
Offline RL trades potential optimality for robustness against 
extrapolation errors 
19 


## Page 20

Policy constraints toward the behavior 
policy 
Policy-constrained methods restrict 𝜋 from deviating far from 
behavior policy 𝛽 
Imitation-style regularization adds objectives that keep 𝜋𝑎
𝑠 
close to 𝛽𝑎
𝑠 
Examples include KL divergence penalties or additional behavior 
cloning losses 
Generative models approximate behavior’s action distribution; 
policies choose among sampled candidate actions 
Hard support constraints forbid low-probability actions or project 
policies back into data support 
20 


## Page 21

Conservative Q-functions and uncertainty-
based pessimism 
Conservative Q-learning (CQL) augments Bellman loss with penalties 
for high Q-values off the dataset 
Objective encourages higher Q-values on in-dataset actions, lower 
values on broad action samples 
Greedy policy then prefers actions well supported by the logged data 
Other methods estimate uncertainty using Q-ensembles or Bayesian-
style approximations 
Effective pessimistic value:  
𝑄pess 𝑠, 𝑎= 𝑄 𝑠, 𝑎−𝜆⋅uncertainty 𝑠, 𝑎 
21 


## Page 22

Offline RL as off-policy RL with principled 
fear 
Offline RL removes exploration, so overestimated values rarely 
receive corrective feedback 
Behavior policy defines the support where data-based estimates are 
trustworthy 
Naïve off-policy Q-learning propagates hallucinated values from 
unsupported regions 
Modern methods constrain policies or adjust Q-values to encourage 
conservative behavior 
Offline RL becomes “off-policy RL with principled fear” of actions 
absent from the dataset 
22 


## Page 23

Safe RL: core tension 
Exploration drives agents toward high reward in unknown 
environments 
Real systems impose “do not crash, melt, or bankrupt” constraints 
Safe RL studies reward hacking, hard safety constraints, and 
robustness 
Focus lies on systems deployed in changing, imperfectly modeled 
worlds 
Safety becomes a first-class design concern, not an afterthought 
23 


## Page 24

Reward hacking and specification gaming: 
definition 
Agents optimize the provided reward, not designers’ intentions 
Misaligned reward functions invite clever shortcuts and loopholes 
Reward hacking or specification gaming describes this behavior 
Optimization targets proxy metrics rather than true human goals 
Goodhart’s law: targeted measures stop reflecting the underlying 
objective 
24 


## Page 25

Reward hacking: gaming proxy rewards 
CoastRunners agent maximized points by circling respawning targets 
in a lagoon 
Finishing the race became irrelevant; score already maximized 
DeepMind racing agent spun around green blocks for shaping reward 
In both cases, proxies stopped tracking “win the race” 
Optimization faithfully followed reward while violating designers’ 
intent 
25 


## Page 26

Reward hacking in safety-critical domains 
Recommendation systems maximizing clicks may promote harmful 
or sensational content 
Trading agents maximizing profit without risk terms can take 
catastrophic bets 
Robots with sparse success rewards may slam into walls without 
collision penalties 
Safe RL treats such failures as central, not anecdotal 
Two questions emerge: expressing constraints and handling 
distribution shift 
26 


## Page 27

CMDPs: rewards, costs, and safety budgets 
Constrained MDPs split performance and safety into reward and cost 
At time t: reward rt, cost ct for unsafe or resource usage 
Example 1: ct = 1 on collision, otherwise 0 
Example 2: ct equals energy consumption or distance to humans 
Optimization problem:  
max
𝜋𝐸
 𝑟𝑡
𝑡
 s.t. 𝐸 𝑐𝑡
𝑡
≤𝑑 
27 


## Page 28

Interpreting costs and budgets 
Reward measures task performance quality 
Cost measures safety budget usage or risk exposure 
Budget d encodes maximum acceptable expected cumulative cost 
Costs can represent forbidden regions, constraint violation counts, or 
risk measures 
Framework’s usefulness depends heavily on well-chosen cost 
definitions 
28 


## Page 29

Lagrangian methods for safe RL 
Lagrangian approach converts constraint into a single unconstrained 
objective 
Objective: 
𝐸 𝑟𝑡
𝑡
𝜆𝐸 𝑐𝑡
𝑡
−𝑑 
Multiplier 𝜆 penalizes policies exceeding the safety budget 
Algorithms update policy and 𝜆 together during learning 
Safety appears as a separate channel, not folded into reward 
29 


## Page 30

Shielding, safe sets, and action filters 
Shielding methods learn or specify a safe state set 
Exploration and control remain inside this safe region 
Action filters intercept proposed actions before execution 
Unsafe actions are modified or replaced to satisfy constraints 
Constraints apply during training and deployment, though emphasis 
may differ 
30 


## Page 31

Robustness: distribution shift in RL 
Deployment environment often differs from the training 
environment 
Dynamics can change: friction, masses, delays, or contact properties 
Observations can change: sensor degradation, lighting, new obstacles 
Other agents can adapt, altering interaction patterns in multi-agent 
settings 
Policy-induced trajectories may differ from behavior policy 
trajectories, causing additional shift 
31 


## Page 32

Robust RL strategies and the three lenses 
Domain randomization trains across varied parameters to improve 
transfer 
Adversarial training introduces worst-case perturbations within 
bounded sets 
Distributionally robust objectives optimize worst-case performance 
over nearby distributions 
Safe RL unifies three lenses: reward hacking, constraints, and 
robustness 
Overall goal: clear objectives, explicit safety limits, and resilience 
when the world shifts 
32 


## Page 33

Hierarchical RL: idea and motivation 
HRL breaks big tasks into smaller skills and stitches skills into full 
solutions 
Flat RL chooses primitive actions every step, for example “move 
north/south/east/west” 
Long horizons make flat policies slow to learn and fragile to credit 
assignment 
HRL separates high-level decisions (“go to elevator”) from low-level 
control (“walk, turn, stop”) 
Goal: reason in skills and subtasks rather than treating every time 
step as identical 
33 


## Page 34

Temporal abstraction and options 
Options generalize actions to multi-step skills 
Each option 𝜔 has an initiation set 𝐼𝜔 of valid starting states 
Intra-option policy 𝜋𝜔𝑎
𝑠 chooses primitive actions while option 
runs 
Termination condition 𝛽𝜔𝑠 gives probability that option stops in 
state s 
Primitive actions are degenerate one-step options that always 
terminate immediately 
34 


## Page 35

Meta-policy and SMDP view 
A meta-policy 𝜇𝜔
𝑠 selects which option to start in each state 
Option value function: 
𝑄𝜇𝑠, 𝜔= expected return from 𝑠 using 𝜔 then 𝜇 
Decisions occur at option boundaries, not every primitive time step 
The resulting process is a semi-Markov decision process (SMDP) 
HRL compresses long trajectories into fewer, semantically 
meaningful jumps 
35 


## Page 36

Benefits and downsides of hierarchy 
Sample efficiency: reuse skills such as “go to door” across episodes 
and tasks 
Long-term credit assignment improves when credit attaches to 
subgoals like “get key” 
Structured exploration uses “try a different skill” instead of random 
primitive noise 
Transfer: skills like “grasp object” or “stand up” move across tasks 
and environments 
Downsides: discovering good subgoals is hard and fixed hierarchies 
can be suboptimal 
36 


## Page 37

Subgoal discovery strategies 
Bottleneck states: door cells or chokepoints lie on many successful 
trajectories 
Graph analyses identify high “betweenness” states and declare them 
subgoals 
State abstraction and clustering define regions; boundaries between 
regions become subgoals 
Intrinsic motivation rewards novel or interesting states, so consistent 
events become subgoals 
Goal-conditioned values V(s, g) (UVFA) treat goals as inputs and 
yield goal-conditioned skills 
37 


## Page 38

MAXQ: value decomposition by task tree 
MAXQ starts from a human-designed task hierarchy, not explicit 
options 
Root task solves the full problem, for example “deliver passenger” in 
Taxi 
Subtasks handle pieces: “navigate to passenger”, “pick up”, “navigate 
to destination”, “drop off” 
Each subtask is an MDP with its own termination condition and value 
function 
Global value splits into current subtask value plus value after subtask 
completion, enabling reuse 
38 


## Page 39

HAMs and bottleneck-based options 
Hierarchical Abstract Machines (HAMs) encode behavior as finite-
state controllers 
Nodes represent modes like “search corridor” or “search key”; edges 
specify transitions 
Some nodes output primitive actions, others invoke lower-level 
machines 
Bottleneck-based methods analyze random trajectories to find states 
lying “between” many others 
These bottlenecks become subgoals; corresponding options learn “go 
to bottleneck i” 
39 


## Page 40

Feudal networks and h-DQN 
Feudal networks use a manager–worker separation 
Manager observes abstract representations and outputs goal vectors 
in latent space 
Workers receive state and goal vector, then output primitive actions 
to move toward that goal 
Hierarchical DQN (h-DQN) uses a meta-controller choosing subgoal g 
and a goal-conditioned DQN 
Controller gets intrinsic reward for achieving g; meta-controller gets 
environment reward 
40 


## Page 41

HIRO, Option-Critic, and HAC 
HIRO sets continuous subgoal states for a lower-level policy 
Replay relabels high-level actions with subgoals actually achieved, 
stabilizing off-policy learning 
Option-Critic parameterizes intra-option policies, terminations, and 
option-selection with neural networks 
Hierarchical Actor-Critic (HAC) stacks actor–critic levels, each setting 
state goals for the lower level 
HAC uses hindsight relabeling and subgoal testing penalties to handle 
non-stationarity between levels 
41 


## Page 42

Environments and mental model 
Four-room grid worlds: doors as subgoals, options as “go to door i” 
skills 
Montezuma’s Revenge: subgoals like “get key” or “reach door” solve 
sparse reward exploration 
Robotic manipulation and locomotion: reusable skills for “reach”, 
“pick”, “place”, “open drawer” 
Multi-agent domains: team strategies on top, individual tactics below 
fit hierarchical control 
Big picture: HRL replaces direct state-to-action mapping with 
reasoning over subgoals, options, and skills 
42 


## Page 43

Conclusions 

Imitation learning and RLHF steer agents toward human-aligned 
behavior; imitation copies actions, RLHF optimizes learned preference-
based reward signals 

Offline RL learns policies from fixed logged data, and enforces 
pessimism for unsupported actions to avoid harmful extrapolation 

Safe RL formalizes constraints and costs, tackles reward hacking, and 
seeks robust performance under distribution shift 

Hierarchical RL introduces temporal abstraction through skills and 
subgoals, improving exploration, sample efficiency, and long-horizon 
credit assignment 
43 
