# RL10_ModernAC

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL10_ModernAC.pdf

**Pages:** 73

---


## Page 1

Reinforcement Learning 
10. Modern Actor-Critic Methods 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025, Draft 
 


## Page 2

Modern Actor-Critic Methods 
1. Proximal Policy Optimization (PPO) 
2. Soft Actor-Critic (SAC) 
3. Dreamer 
2 


## Page 3

Modern Actor-Critic Methods 
1. Proximal Policy Optimization (PPO) 
2. Soft Actor-Critic (SAC) 
3. Dreamer 
3 


## Page 4

PPO: Stabilised Actor–Critic 
PPO is a policy gradient method and stabilised actor–critic variant 
It keeps the conceptual simplicity of REINFORCE and basic actor–
critic 
Added mechanism explicitly limits how much the policy changes per 
update 
Restricting updates keeps the new policy close to the current one 
This stability helped PPO become a default deep RL algorithm 
4 


## Page 5

Where PPO Fits Among RL Methods 
Monte Carlo control, Sarsa, Q-learning, DQN emphasize value 
functions and greedy policies 
Policy gradient methods parameterize the policy 𝜋𝜃𝑎
𝑠 directly 
Objective becomes maximizing expected return by adjusting 
parameters 𝜃 
A typical gradient sample uses the pair ∇𝜃log 𝜋𝜃
𝑎𝑡
𝑠𝑡, 𝐴𝑡
  
𝐴𝑡
  estimates the advantage of action 𝑎𝑡 in state 𝑠𝑡 
5 


## Page 6

Actor–Critic Instability and PPO’s Goal 
Actor–critic methods add a learned value function to reduce gradient 
variance 
Single gradient steps may still change the policy very aggressively 
Old trajectories then no longer describe behaviour of the updated 
policy well 
This mismatch can create strong instability and even complete 
learning collapse 
PPO modifies the update rule to keep each policy change within a 
safe region 
6 


## Page 7

Policy Ratios in PPO 
PPO compares old and new policies with a probability ratio 
Definition: 
𝑟𝑡𝜃=
𝜋𝜃𝑎𝑡
𝑠𝑡
𝜋𝜃old 𝑎𝑡
𝑠𝑡
 
𝑟𝑡≈1 means little change; 𝑟𝑡> 1 increases, 𝑟𝑡< 1 decreases action 
probability 
The same ratio appears in off-policy Monte Carlo and importance 
sampling 
PPO reuses 𝑟𝑡 to measure how aggressive each policy update is on 
sampled data 
7 


## Page 8

Surrogate Objective in Policy Gradient 
Basic policy gradient objective: 
𝐿𝑃𝐺𝜃= 𝔼t log 𝜋𝜃𝑎𝑡
𝑠𝑡
, 𝐴𝑡
  
Gradient increases probability of actions with positive 𝐴𝑡
  
Gradient decreases probability of actions with negative 𝐴𝑡
  
Actor–critic versions still follow this basic objective form 
8 


## Page 9

Conservative Policy Iteration Form 
Rewrite with probability ratios: 
𝑟𝑡𝜃= 𝜋𝜃𝑎𝑡
𝑠𝑡
𝜋𝜃𝑜𝑙𝑑𝑎𝑡
𝑠𝑡
 
Surrogate objective: 
𝐿𝐶𝑃𝐼𝜃= 𝔼𝑡𝑟𝑡𝜃, 𝐴𝑡
  
Intuition: increase 𝑟𝑡 when 𝐴𝑡
 > 0, decrease when 𝐴𝑡
 < 0 
Large 𝑟𝑡 or small 𝑟𝑡 mean the new policy moved far from the data 
policy 
9 


## Page 10

Instability and TRPO’s Trust Region Idea 
Repeated gradient steps on 𝐿𝐶𝑃𝐼 can push 𝜃 too far 
Ratios 𝑟𝑡 can become extreme; old samples no longer match the new 
policy 
Learning then becomes unreliable and can collapse completely 
TRPO constrains KL divergence between old and new policies 
TRPO enforces a trust region but requires complex constrained 
optimization 
10 


## Page 11

PPO’s Clipped Surrogate Objective 
PPO mimics a trust region with a simple clipped objective 
Clipped loss: 
𝐿𝑐𝑙𝑖𝑝𝜃= 𝔼𝑡min 𝑟𝑡𝜃𝐴𝑡
 , clip 𝑟𝑡𝜃, 1 −𝜖, 1 + 𝜖𝐴𝑡
 
 
Clip keeps 𝑟𝑡𝜃 in 1 −𝜖, 1 + 𝜖 inside the loss 
𝜖 is small, typically (0.1) or (0.2) 
11 


## Page 12

Clipping Behaviour for Positive and 
Negative Advantage 
Case 𝐴𝑡
 > 0: increasing 𝑟𝑡 above 1 helps until 1 + 𝜖 
Once 𝑟𝑡> 1 + 𝜖, the clipped term stops growing; extra increase 
brings no benefit 
Case 𝐴𝑡
 < 0: decreasing 𝑟𝑡 below 1 helps until 1 −𝜖 
Once 𝑟𝑡< 1 −𝜖, the clipped term stops decreasing; further 
reduction gives no gain 
12 


## Page 13

Soft Trust Region Intuition 
PPO allows 𝑟𝑡 to move away from 1 inside 1 −𝜖, 1 + 𝜖 
Inside this interval, learning behaves like a standard policy gradient 
method 
Outside the interval, the objective discourages further change on 
those samples 
Gradients naturally pull updates back toward smaller, safer policy 
shifts 
Clipping implements a soft trust region in the loss without second-
order methods 
13 


## Page 14

PPO’s Combined Objective 
PPO optimizes policy, value function, and entropy together 
Total loss: 
 
𝐿𝑡𝑜𝑡𝑎𝑙𝜃, 𝜙= 𝔼t 𝐿𝑐𝑙𝑖𝑝𝜃+ 𝑐1𝐿𝑉𝐹𝜙+ 𝑐2𝑆𝜋𝜃⋅𝑠𝑡
 
 
𝜃: policy parameters; 𝜙: value function parameters 
𝐿𝑉𝐹𝑡: value loss for 𝑉𝜙𝑠𝑡 
𝑆𝜋𝜃⋅𝑠𝑡
: entropy of the policy at state 𝑠𝑡 
14 


## Page 15

Value Loss and Entropy Bonus 
Value loss pushes 𝑉𝜙𝑠𝑡 toward an empirical return 𝑅𝑡 
Usually a squared error: 𝑉𝜙𝑠𝑡−𝑅𝑡
2 
Learned value acts as a baseline and reduces variance of 𝐴𝑡
  
Entropy term encourages high-entropy (more random) policies 
Coefficients 𝑐1, 𝑐2  balance policy improvement, value accuracy, and 
exploration 
15 


## Page 16

Advantage Estimation with GAE 
PPO uses an advantage estimate 𝐴𝑡
  rather than raw returns 
Temporal-difference error: 
𝛿𝑡= 𝑟𝑡+ 𝛾𝑉𝜙𝑠𝑡+1 −𝑉𝜙𝑠𝑡 
Generalized Advantage Estimation (GAE): 
𝐴𝑡
 = 𝛿𝑡+ 𝛾𝜆𝛿𝑡+1 𝛾𝜆2𝛿𝑡+2 + ⋯ 
Combines TD errors over future steps with geometric weights 
Extends ideas from TD(𝜆) and eligibility traces 
16 


## Page 17

GAE: Bias–Variance Trade-off and Intuition 
𝜆≈1: long-horizon estimate, low bias, high variance 
Smaller 𝜆: more reliance on short-term TD errors, higher bias, lower 
variance 
GAE measures how actual continuation differs from value-function 
expectations 
Positive 𝐴𝑡
  signals surprisingly good outcomes from 𝑠𝑡 
This surprise signal directly shapes PPO’s clipped policy update 
17 


## Page 18

PPO Training Loop: Data Collection and 
Estimation 
Current stochastic policy runs in the environment for several time 
steps 
Often many parallel actors collect trajectories simultaneously 
Collected data: states, actions, rewards, and value predictions 
Returns and advantages are computed for each time step using GAE 
These trajectories form a single “batch” for the next optimisation 
phase 
18 


## Page 19

PPO Training Loop: Multiple Optimisation 
Epochs 
PPO reuses the same batch for several epochs of mini-batch SGD or 
Adam 
Policy in the denominator, 𝜋𝜃𝑜𝑙𝑑, stays fixed during these epochs 
Only the numerator policy 𝜋𝜃 changes while optimizing 𝐿𝑐𝑙𝑖𝑝 
Clipping ensures policy ratios cannot wander too far from 1 on this 
data 
Many epochs still produce conservative changes because extreme 
ratios stop improving the objective 
19 


## Page 20

PPO Training Loop: Value Updates and On-
Policy Nature 
Value function parameters 𝜙 update using the same batch of 
trajectories 
Value loss uses a separate gradient step, often with its own optimizer 
After optimization, the algorithm sets 𝜃𝑜𝑙𝑑←𝜃 
New trajectories are then collected with the updated policy 
PPO remains on-policy; it does not rely on experience replay buffers 
20 


## Page 21

KL-Penalty PPO Variant 
Original paper also explored a KL-penalty version of PPO 
This variant adds a KL divergence term between old and new policies 
into the objective 
Algorithm tracks actual KL and compares it with a target KL value 
Penalty coefficient increases or decreases to steer KL toward the 
target 
Behaviour mimics an adaptive trust region but proved harder to tune 
than clipping 
21 


## Page 22

Distributed PPO (DPPO) 
DeepMind developed a distributed PPO implementation for large-
scale experiments 
Many workers collect trajectories in parallel under a shared policy 
Gradients from workers are synchronised to update a central set of 
parameters 
Objective still follows PPO with regularised policy and value losses 
Results on locomotion tasks show complex skills for walkers, 
quadrupeds, and humanoids 
22 


## Page 23

Why PPO Works: Stability and Sample 
Efficiency 
Clipped objective 𝐿𝑐𝑙𝑖𝑝 discourages destructive, overly large policy 
updates 
Once ratios leave the trust region, further change on those samples 
brings no benefit 
Multiple epochs over each batch improve sample efficiency for an on-
policy method 
Each trajectory contributes many gradient steps instead of a single 
noisy update 
REINFORCE-style methods discard data after one update and 
therefore waste information 
23 


## Page 24

Why PPO Works: Simplicity and Robustness 
Implementation mainly changes the loss in a standard actor–critic 
network 
No second-order optimisation, conjugate gradients, or line searches 
are required 
Reference implementations, such as Spinning Up, closely match this 
simple structure 
PPO works reliably across many tasks with reasonable default 
hyperparameters 
Empirical studies show performance competitive with or better than 
TRPO, A2C, and others 
24 


## Page 25

Conceptual Summary of PPO 
PPO is actor–critic with a self-protecting, ratio-based surrogate loss 
Policy update uses 𝑟𝑡𝜃𝐴𝑡
  but clips benefits when ratios move too 
far 
Critic provides value estimates and TD errors that feed GAE for 𝐴𝑡
  
Entropy bonus prevents premature collapse to deterministic policies 
and maintains exploration 
Trust-region intuition is encoded directly in the loss, yielding a 
simple, reliable deep RL algorithm 
25 


## Page 26

Modern Actor-Critic Methods 
1. Proximal Policy Optimization (PPO) 
2. Soft Actor-Critic (SAC) 
3. Dreamer 
26 


## Page 27

SAC: High-Level Picture 
SAC is an off-policy actor–critic algorithm for continuous control 
Combines value-based off-policy learning with stochastic policy 
gradients 
Uses a maximum-entropy objective that explicitly rewards 
“purposeful randomness” 
Aims for stability, sample efficiency, and robustness across tasks 
Replaces brittle deterministic behavior with deliberately noisy 
policies 
27 


## Page 28

From DDPG/TD3 to SAC 
DDPG and TD3 optimise deterministic policies for expected return 
only 
They achieve strong sample efficiency but often explore poorly 
Deterministic policies over-trust noisy Q-values and get stuck in bad 
optima 
SAC keeps off-policy efficiency but changes the exploration story 
Policy remains noisy as long as entropy does not hurt reward too 
much 
28 


## Page 29

Classic RL Objective 
Standard episodic RL maximizes expected return: 
 
𝐽𝑐𝑙𝑎𝑠𝑠𝑖𝑐𝜋= 𝔼 𝑟𝑠𝑡, 𝑎𝑡
𝑡
 
 
Optimal policy in fully observed MDPs tends toward determinism 
Policy eventually commits to the single best action per state 
Determinism is risky with noisy, approximate value functions 
Small estimation errors can produce fragile, over-committed 
behavior 
29 


## Page 30

Maximum Entropy Objective 
Maximum entropy RL augments the reward with an entropy bonus 
Objective: 
𝐽𝑚𝑎𝑥−𝑒𝑛𝑡𝜋= 𝐸 𝑟𝑠𝑡, 𝑎𝑡+ 𝛼, ℋ𝜋⋅𝑠𝑡
𝑡
 
Policy entropy: 
 
 trades off reward versus entropy 
As 𝛼→0, objective reduces to standard RL 
30 


## Page 31

Consequences of the Max-Entropy View 
High entropy keeps multiple promising actions alive per state 
Exploration improves because randomness is part of the objective 
Robustness increases; policy avoids over-commitment to narrow 
strategies 
Multiple behavioral modes can coexist when actions are similarly 
good 
For finite 𝛼, the optimal policy is intentionally stochastic 
31 


## Page 32

SAC as “Soft” Actor–Critic 
SAC follows the usual actor–critic pattern with modifications 
Uses off-policy data and replay-style updates like DDPG/TD3 
Policy update targets high soft Q-values, not only plain Q-values 
Soft Q-values incorporate both reward and future entropy 
Framework formalises the trade-off between performance and 
purposeful randomness 
32 


## Page 33

Main Function Approximators in SAC 
Stochastic policy 𝜋𝜙𝑎
𝑠 serves as the actor 
Policy often modeled as a Gaussian with neural-network mean and 
standard deviation 
Two Q-networks 𝑄𝜃1 𝑠, 𝑎 and 𝑄𝜃2 𝑠, 𝑎 serve as critics 
Dual critics help control overestimation bias, echoing Double Q ideas 
A pair of target Q-networks provides delayed, stable bootstrapping 
targets 
33 


## Page 34

Value Network and Standard SAC Variant 
Original SAC formulation used a separate value network V(s) 
Later simplification removed V(s) and relied directly on twin Q-
functions 
Modern implementations usually follow this streamlined “standard 
variant.” 
Overall flow still mirrors DDPG/TD3 structurally: actor proposes, 
critics evaluate 
Key change: actor is trained to maximize soft Q, not solely expected 
reward 
34 


## Page 35

Soft Q-Values and the Soft Bellman Backup 
Standard Q-learning backup assumes greedy next actions: 
𝑄𝑠, 𝑎≈𝑟𝑠, 𝑎+ 𝛾𝔼𝑠′ max 𝑎′ 𝑄𝑠′, 𝑎′
 
Maximum entropy RL replaces greedy choice with sampling from an 
optimal stochastic policy 
Soft backup uses a soft value: 
 
 
The term −𝛼log 𝜋𝑎′ 𝑠′  injects the entropy bonus at the next step 
High soft Q(s,a) indicates actions with good reward and beneficial 
future stochasticity 
35 


## Page 36

Double Critics and Soft Targets in SAC 
Deep methods often overestimate Q-values when maximization 
appears inside updates 
SAC maintains two critics and uses the minimum of their target 
predictions 
Soft target for transition (s, a, r, s'): 
𝑦= 𝑟+ 𝛾min
𝑖=1,2 𝑄𝜃𝑖𝑠′, 𝑎′ −𝛼log 𝜋𝜙𝑎′ 𝑠′
 
with 𝑎′ ∼𝜋𝜙⋅𝑠′  
Target networks 𝑄𝜃𝑖 change slowly and keep bootstrapping stable 
Critics learn expected future reward plus entropy under the current 
stochastic policy, using conservative targets 
36 


## Page 37

Soft Policy Improvement in SAC 
Standard actor–critic increases expected Q(s,a) under the policy 
SAC still prefers high-Q actions but also values entropy 
Good actions balance large Q with sufficient randomness 
Policy improvement trades reward against uncertainty, not reward 
alone 
Objective explicitly encodes this Q–entropy trade-off 
37 


## Page 38

KL View and Ideal Max-Entropy Policy 
Ideal maximum-entropy policy for a given Q: 
𝜋⋆𝑎
𝑠
∝exp 1
𝛼𝑄𝑠, 𝑎
 
High-Q actions receive high probability but other actions never 
vanish completely 
Temperature 𝛼 controls how sharp or diffuse preferences are 
SAC cannot represent 𝜋⋆ exactly within restricted policy classes 
Actor instead moves toward 𝜋⋆ by minimizing a KL divergence 
38 


## Page 39

Practical Actor Objective and Interpretation 
SAC actor objective: 
𝐽𝑎𝑐𝑡𝑜𝑟𝜙= 𝔼𝑠∼𝒟,𝑎∼𝜋𝜙𝛼log 𝜋𝜙
𝑎
𝑠
−𝑄𝜃𝑠, 𝑎 
The −𝑄𝜃𝑠, 𝑎 term discourages low-value actions 
The 𝛼log 𝜋𝜙
𝑎
𝑠 term penalizes overly peaked distributions 
Minimizing 𝐽𝑎𝑐𝑡𝑜𝑟 puts mass on high-Q actions while maintaining 
entropy 
Actor approximates the exponentiated-Q distribution within its 
parameterization 
39 


## Page 40

Gaussian Actor and Reparameterisation 
Policy usually Gaussian with neural-network mean and log standard 
deviation 
Actions sampled via reparameterization: 
𝑎= tanh 𝜇𝜙𝑠+ 𝜎𝜙𝑠, 𝜉,  𝜉∼𝒩0, 𝐼 
Reparameterization keeps sampling differentiable for 
backpropagation 
Tanh squashing enforces bounded actions matching environment 
constraints 
Actor network learns both mean behavior and exploration scale 
40 


## Page 41

Temperature and Reward Scaling 
Temperature 𝛼 sets trade-off between reward and entropy 
Large 𝛼 : strong entropy bonus, very random policies 
Small 𝛼: entropy suppressed, policy approaches deterministic 
behavior 
In max-entropy RL, reward rescaling changes the optimal policy 
unless 𝛼 adjusts 
Choosing 𝛼 well is crucial for good performance 
41 


## Page 42

Automatic Temperature Tuning in SAC 
SAC treats 𝛼 as a learnable parameter, not a fixed hyperparameter 
Conceptually solves: maximize return subject to entropy above a 
target 
Lagrangian introduces 𝛼 as dual variable for the entropy constraint 
Practical loss: 
𝐽𝛼= 𝐸𝑠,𝑎∼𝜋𝜙−𝛼log 𝜋𝜙
𝑎
𝑠
+ ℋ𝓉𝒶𝓇ℊℯ𝓉
 
Low entropy (below target) pushes 𝛼 up; high entropy pushes 𝛼 
down, adapting exploration automatically 
42 


## Page 43

SAC Training Loop: Interaction and Replay 
SAC stores transitions (s, a, r, s') from the current stochastic policy in 
a replay buffer 
Environment interaction may use a single agent or many parallel 
actors 
Replay buffer keeps experience from recent policies, not only the 
latest one 
Off-policy design permits learning from older data and varied 
behavior 
Robotics settings benefit because physical samples arrive slowly and 
expensively 
43 


## Page 44

SAC Training Loop: Gradient Updates 
Mini-batches sampled from replay drive several gradient steps per 
environment phase 
Critics minimise soft Bellman error using the conservative double-Q 
target 
Actor minimises the soft objective mixing Q-values and log-
probabilities 
Temperature parameter updates toward a target entropy through its 
own loss 
Target Q-networks track critics with an exponential moving average 
update 
44 


## Page 45

Off-Policy Efficiency and Real-World 
Example 
Off-policy replay makes old experience useful for many updates 
Sample efficiency significantly exceeds that of on-policy methods like 
PPO 
Efficiency becomes crucial in real robots with wear-and-tear and 
reset costs 
Minitaur experiments show SAC learning robust quadruped gaits on 
hardware within hours 
Learned policies tolerate perturbations and terrain variations 
without catastrophic failure 
45 


## Page 46

Why SAC Works Well in Practice 
Maximum-entropy objective encourages broad, persistent 
exploration 
Stochastic policies remain less brittle to modelling errors and 
dynamics shifts 
Twin critics with conservative targets prevent severe overestimation 
and instability 
Off-policy formulation reuses data heavily while retaining a flexible 
stochastic actor 
Automatic temperature tuning adapts exploration level across 
training stages 
46 


## Page 47

Practical Advantages and Conceptual 
Summary 
SAC relies on first-order gradients and standard neural network 
components 
Implementation complexity matches DDPG or TD3, simpler than 
trust-region methods 
Single hyperparameter set often performs well across diverse 
continuous-control tasks 
SAC resembles Q-learning in a maximum-entropy world plus an 
exponentiated-Q actor 
Compared with PPO, SAC suits continuous actions and offers 
stronger sample efficiency 
47 


## Page 48

Modern Actor-Critic Methods 
1. Proximal Policy Optimization (PPO) 
2. Soft Actor-Critic (SAC) 
3. Dreamer 
48 


## Page 49

Dreamer: Model-Based “Thinking Ahead” 
Dreamer is a model-based deep RL algorithm with a learned world 
model 
DreamerV3 uses one configuration across hundreds of tasks without 
per-domain tuning 
Control problem: act by “thinking ahead” in the learned model, not 
just reacting 
PPO and SAC learn directly from real transitions; Dreamer first 
learns an internal simulator 
Separation: “understand the world” first, then “decide what to do” 
via imagined rollouts 
49 


## Page 50

From Model-Free RL to Latent World 
Models 
Model-free methods (DQN, PPO, SAC) use the real environment for 
all next states 
Every gradient step needs fresh experience, which leads to sample 
hunger 
No way to ask “what if I tried this action elsewhere?” without 
executing it 
Dreamer follows Dyna: learn a model, then use it for planning and 
policy learning 
Model predicts latent dynamics instead of raw pixels or sensor 
streams 
50 


## Page 51

Latent Dynamics Model: Core Components 
Encoder compresses each observation into a compact latent state 
Recurrent state-space model predicts how the latent state evolves 
under actions 
Decoders map latent states back to observations 
Decoders also reconstruct rewards and continuation (episode) flags 
Planning and imagination occur entirely in this low-dimensional 
latent space 
51 


## Page 52

Why Plan in Latent Space? 
Agent never imagines raw images, only abstract latent states 
Latent states summarise the aspects of the observation that matter 
for control 
Latent planning avoids the curse of dimensionality of pixel-level 
prediction 
Imagined trajectories become cheap to generate once the model is 
trained 
Real experience mainly serves to refine this latent dynamics model 
52 


## Page 53

World Model Structure in Dreamer 
Architecture: world model at the bottom, actor–critic module on top 
The latent state 𝑠𝑡 splits into deterministic recurrent state ℎ𝑡 and 
stochastic 𝑧𝑡 
The encoder maps observation 𝑥𝑡 into stochastic latent variable 𝑧𝑡 
The recurrent part updates hidden ℎ𝑡 from previous hidden state and 
action 
The pair ℎ𝑡, 𝑧𝑡 defines the full model state 𝑠𝑡 
53 


## Page 54

Predictions from the Model State 
From 𝑠𝑡, the model predicts the reward at time t 
Model predicts whether the episode continues or terminates 
Model reconstructs the original observation 𝑥𝑡 
Dynamics predictor forecasts next latent 𝑧𝑡+1 from current hidden 
state and action 
Recurrent state-space model rolls forward without seeing new 
observations 
54 


## Page 55

Discrete Latents and Straight-Through 
Training 
DreamerV3 represents each coordinate of 𝑧𝑡 as a small categorical 
distribution 
Joint latent forms a vector of categorical variables rather than a 
Gaussian 
Straight-through gradients make sampling appear discrete while 
gradients remain continuous 
Backpropagation treats discrete choices as differentiable 
approximations 
Discrete latents provide greater stability than fully continuous 
Gaussian latents 
55 


## Page 56

Three Losses for Learning the World Model 
World model trains on replayed experience using multiple loss terms 
Prediction loss encourages accurate reconstructions of observations, 
rewards, and continuation 
Dynamics loss encourages forecasting of 𝑧𝑡+1 from current hidden 
state 
Representation loss encourages latents that dynamics can predict 
reliably 
Combined objective yields informative and predictable latent 
representations 
56 


## Page 57

Posterior–Prior KL and Free Bits 
Dynamics and representation losses use KL divergence between 
posterior and prior 
Posterior: encoder distribution conditioned on the current 
observation 
Prior: dynamics predictor distribution conditioned only on past states 
and actions 
DreamerV3 applies “free bits”, clipping KL terms below a threshold 
Free bits prevent collapse into trivial, low-information latents that 
ignore inputs 
57 


## Page 58

Regularised Discrete Latents and 
Imagination 
Small amount of uniform noise mixes into categorical distributions 
for 𝑧𝑡 
Noise stops latents from becoming perfectly deterministic and avoids 
KL spikes 
Trained world model can roll forward from an initial latent plus an 
action sequence 
Imagined trajectories evolve entirely in latent space without new 
visual input 
Long-range video predictions (mazes, walking robots) illustrate this 
internal world model 
58 


## Page 59

Handling Scale: Symlog and Symexp 
DreamerV3 must handle tiny and huge rewards, short and long 
horizons 
Raw squared losses misbehave when reward scales differ by orders 
of magnitude 
Symlog transform: 
𝑠𝑦𝑚𝑙𝑜𝑔𝑥= 𝑠𝑖𝑔𝑛𝑥 log 𝑥+ 1  
Large magnitudes compress; small values near zero stay almost 
unchanged 
Rewards, returns, even observations pass through symlog to keep 
gradients bounded 
59 


## Page 60

Inverting the Transform and Two-Hot 
Regression 
Symexp is the inverse mapping back to original scale: 
𝑠𝑦𝑚𝑒𝑥𝑝𝑦= 𝑠𝑖𝑔𝑛𝑦
exp 𝑦
−1  
Dreamer predicts noisy scalars via distributions, not direct 
regression 
Network outputs logits over exponentially spaced scalar bins 
Two-hot targets split weight between the two nearest bins with 
interpolation 
Cross-entropy compares soft targets and predictions, focusing 
gradients on probability mass shifts 
60 


## Page 61

Imagination Rollouts in Latent Space 
After training, the world model compresses and predicts real 
experience 
Actor and model start from latent states linked to replayed 
observations 
Actor samples an action from its policy given the current latent state 
World model predicts next latent state, reward, and continuation flag 
Repeated application yields imagined trajectories of roughly 16 latent 
steps 
61 


## Page 62

Latent Distributional Critic and λ-Returns 
Critic consumes imagined rewards and latent states from these 
rollouts 
It predicts a return distribution using categorical bins, spaced 
exponentially 
Dreamer applies distributional value prediction instead of scalar 
values 
λ-returns mix multi-step returns with bootstrapping from later value 
estimates 
This combination stabilizes training and propagates information 
across imagined horizons 
62 


## Page 63

Latent Actor Objective and Return 
Normalisation 
Actor seeks actions maximizing return while preserving policy 
entropy 
Actor gradient resembles REINFORCE: log-probabilities weighted by 
return-like signals 
Returns are normalized using within-batch percentiles, not standard 
advantage normalization 
Exponential moving average tracks return range, keeping effective 
scale near [0, 1] 
Single entropy coefficient then works across sparse and dense 
reward domains 
63 


## Page 64

World Model as the Actor–Critic’s 
Environment 
Actor and critic both operate purely on latent states 𝑠𝑡= ℎ𝑡, 𝑧𝑡 
From their perspective, the world model is the environment 
Real environment intervenes only during fresh data collection 
Imagined rollouts supply most gradients for policy and value 
learning 
Separation of modelling and control enables broad generalization 
across tasks 
64 


## Page 65

DreamerV3 Training Loop: Real Interaction 
Current actor interacts with the real environment for several steps 
Observations arrive and pass through the encoder into latent states 
Actor samples actions based on these latent states, not raw pixels 
True rewards and next observations are stored in a replay buffer 
Replay buffer holds sequences that later train both world model and 
critic 
65 


## Page 66

Training the World Model from Replay 
Algorithm samples sequences of experience from the replay buffer 
Observations are encoded into latents for each time step in the 
sequence 
Recurrent model rolls forward using logged actions and latent states 
Model reconstructs observations, predicts rewards, and predicts 
continuation flags 
Encoder, recurrent core, decoders, and reward/continue heads 
update via combined losses 
66 


## Page 67

Imagination Rollouts and Latent Actor–
Critic 
Dreamer selects latent states at sequence ends as starting points for 
imagination 
World model and actor simulate future steps entirely in latent space 
Reward head provides imagined rewards that define imagined return 
sequences 
Critic receives λ-returns from imagined rollouts as training targets 
Actor receives policy gradients based on normalised returns and an 
entropy bonus 
67 


## Page 68

Critic Anchoring and Self-Consistent 
Universe 
Critic sometimes also trains on real latent trajectories from replay 
Real trajectories anchor value estimates directly to actual rewards 
Critic parameters move toward an exponential moving average of 
themselves 
This regularisation acts as a soft target-network mechanism 
World model predicts, critic evaluates, and actor chooses, forming a 
self-consistent universe 
68 


## Page 69

DreamerV3 Robustness Across Domains 
Single hyperparameter setting works across more than 150 tasks 
Benchmarks include Atari, DeepMind Lab, ProcGen, control suites, 
BSuite, and Minecraft 
KL balancing with free bits prevents model collapse in simple and 
complex environments 
Symlog, symexp, and two-hot losses normalise signal scales 
consistently 
Percentile-based return normalisation supports one entropy scale for 
sparse and dense rewards 
69 


## Page 70

Further Robustness Mechanisms and 
Minecraft Example 
Distributional critic decouples gradient scale from raw return 
magnitude 
Replay ratio and model size scale performance smoothly without 
retuning 
Ablations show monotonic gains as model size and replay ratio 
increase 
Same configuration learns locomotion, navigation, and visual tasks 
without domain-specific tweaks 
In Minecraft, DreamerV3 discovers diamond tools, outperforming 
competitors stuck at iron tools 
70 


## Page 71

Relation to PPO and SAC 
Compared with PPO, Dreamer uses a model-based path instead of 
clipped on-policy gradients 
PPO relies on large batches of fresh data and discards them quickly 
Dreamer reuses data heavily to train both world model and latent 
actor–critic 
Compared with SAC, Dreamer trades direct soft Q-learning for latent 
imagination rollouts 
71 


## Page 72

Conceptual Summary of Dreamer 
Dreamer promotes “understand the world first, then dream before 
acting” 
Recurrent state-space model compresses experience into latents 
predicting observations, rewards, and continuation 
Symlog, symexp, and two-hot encodings stabilize learning across 
diverse reward scales 
Actor–critic learns from imagined trajectories with normalized 
returns and a fixed entropy bonus 
World models plus careful normalization emerge as a promising 
route toward general deep RL 
72 


## Page 73

Conclusions 
PPO offers stable on-policy learning with clipped updates, strong 
robustness and reliable performance across diverse continuous 
control tasks 
SAC achieves high sample efficiency and strong exploration through 
entropy maximization, excelling on challenging, stochastic 
environments and high-dimensional action spaces 
DreamerV3 uses world models for long-horizon planning, gains 
strong sample efficiency and generalization in complex, partially 
observable domains 
73 
