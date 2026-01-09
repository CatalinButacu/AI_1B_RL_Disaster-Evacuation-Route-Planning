# RL07_Eligibility

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL07_Eligibility.pdf

**Pages:** 37

---


## Page 1

Reinforcement Learning 
7. Eligibility Traces 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

Eligibility Traces 
1. The λ-return 
2. TD(λ) 
3. Sarsa(λ) 
2 


## Page 3

Eligibility Traces 
1. The λ-return 
2. TD(λ) 
3. Sarsa(λ) 
 
 
3 


## Page 4

Eligibility Traces 
Eligibility traces form a widely used mechanism in RL algorithms 
They unify and generalize TD and MC methods 
TD(λ) and many temporal-difference methods, e.g., Q-learning or 
Sarsa, use eligibility traces 
Parameter λ extends methods from MC (λ = 1) to one-step TD (λ = 0) 
4 


## Page 5

n-step Returns and Compound Updates 
n-step return uses the first n rewards plus a discounted value 
estimate at step t + n 
  
𝐺𝑡:𝑡+𝑛= 𝑅𝑡+1 + 𝛾𝑅𝑡+2 + ⋯+ 𝛾𝑛−1𝑅𝑡+𝑛+ 𝛾𝑛𝑣 𝑆𝑡+𝑛, 𝑤𝑡+𝑛−1  
 
0 ≤𝑡≤𝑇−𝑛 
  
Any n-step return forms a valid update target for tabular or 
approximate value learning 
We may update toward a weighted average of several n-step returns 
The weights are positive and sum to 1, e.g.: 
  
1
2 𝐺𝑡:𝑡+2 + 1
2 𝐺𝑡:𝑡+4 
5 
optional 


## Page 6

The λ-return 
The λ-return is defined as: 
𝐺𝑡
𝜆= 1 −𝜆 𝜆𝑛−1𝐺𝑡:𝑡+𝑛
∞
𝑛=1
 
Each 𝐺𝑡:𝑡+𝑛 is an n-step return  
The factor 1 −𝜆𝜆𝑛−1 is the weight of that n-step return 
These weights are all positive and sum to 1 
Small λ: mostly short n-step returns 
More bootstrapping, more bias, less variance 
Large λ: mostly long n-step, MC-like returns 
Less bias, more variance 
6 


## Page 7

Example 
Time steps: t = 0, 1, 2, 3 with S3 terminal 
Discount factor: γ = 1 
Rewards: R1 = 0, R2 = 0, R3 = 1 
Current value estimates: 𝑣 𝑆1 = 0.3, 𝑣 𝑆2 = 0.4, 𝑣 𝑆3 = 0 (terminal) 
We want the λ-return 𝐺0
𝜆 at time t = 0 
7 


## Page 8

Computing the n-step Returns from t = 0 

𝐺𝑡:𝑡+𝑛= 𝑅𝑡+1 + 𝛾𝑅𝑡+2 + ⋯+ 𝛾𝑛−1𝑅𝑡+𝑛+ 𝛾𝑛𝑣 𝑆𝑡+𝑛 
1-step return: 𝐺0:1 = 𝑅1 + 𝑣 𝑆1 = 0 + 0.3 = 0.3 
2-step return: 𝐺0:2 = 𝑅1 + 𝑅2 + 𝑣 𝑆2 = 0 + 0 + 0.4 = 0.4 
3-step return: the episode ends at step 3, so no bootstrap term: 
  
𝐺0:3 = 𝑅1 + 𝑅2 + 𝑅3 = 0 + 0 + 1 = 1.0 

𝐺0:3 is just the full MC return 𝐺0 in this episode 
8 


## Page 9

The Finite-Horizon λ-return 
For episodic tasks, the forward-view λ-return at time t with terminal 
time T can be written as: 
  
𝐺𝑡
𝜆= 1 −𝜆
 𝜆𝑛−1𝐺𝑡:𝑡+𝑛
𝑇−𝑡−1
𝑛=1
+ 𝜆𝑇−𝑡−1𝐺𝑡 
Here, T = 3, t = 0 ⇒ T – t – 1 = 2 

⇒𝐺0
𝜆= 1 −𝜆
𝐺0:1 + 𝜆𝐺0:2 + 𝜆2𝐺0 
and 𝐺0 = 𝐺0:3 = 1.0 
We combine: 
Weight 1 −𝜆 on the 1-step return 
Weight 1 −𝜆𝜆 on the 2-step return 
Weight 𝜆2 on the full MC return 
 
9 


## Page 10

Computing the λ-return 
Let λ = 0.5 
Weights: 
For 𝐺0:1: 1 −𝜆=  0.5 
For 𝐺0:2: 1 −𝜆𝜆= 0.5 ⋅0.5 = 0.25 
For 𝐺0: 𝜆2 = 0.25 

⇒𝐺0
0.5= 0.5 ⋅𝐺0:1 + 0.25 ⋅𝐺0:2 + 0.25 ⋅𝐺0  

𝐺0
0.5 = 0.5 ⋅0.3 + 0.25 ⋅0.4 + 0.25 ⋅1 = 0.5 
The extremes: 
λ = 0 → 𝐺0
0 = 𝐺0:1 = 0.3 (pure one-step TD) 
λ = 1 → 𝐺0
1 = 𝐺0 = 1 (pure MC) 
10 


## Page 11

Backup Diagram for TD(λ) 
11 


## Page 12

The λ-return as a Geometric Average 
TD(λ) defines a compound update averaging all n-step returns using 
𝜆∈0,1  
Each 𝐺𝑡:𝑡+𝑛 receives a weight proportional to 𝜆𝑛−1; the factor 1 −𝜆 
normalizes the weights (𝐺𝑡
𝜆= 1 −𝜆 
𝜆𝑛−1𝐺𝑡:𝑡+𝑛
∞
𝑛=1
) 
Weights form a geometric sequence: 1 −𝜆, 1 −𝜆𝜆, 1 −𝜆𝜆2, … 
After termination, all later n-step returns equal 𝐺𝑡, yielding an 
equivalent finite-sum form: 𝐺𝑡
𝜆= 1 −𝜆 
𝜆𝑛−1𝐺𝑡:𝑡+𝑛
𝑇−𝑡−1
𝑛=1
+ 𝜆𝑇−𝑡−1𝐺𝑡 
At λ = 1, λ-return gives an MC update; at λ = 0, one-step TD 
12 


## Page 13

Return Weights 
Weighting given in the λ-return to each of the n-step returns 
13 


## Page 14

The Off-line λ-return Algorithm 
The off-line λ-return algorithm keeps the weight vector unchanged 
during each episode 
After the episode ends, semi-gradient updates are applied for all time 
steps 𝑡= 0, … , 𝑇−1: 
𝑤𝑡+1 = 𝑤𝑡+ 𝛼𝐺𝑡
𝜆−𝑣 𝑆𝑡, 𝑤𝑡
∇𝑣 𝑆𝑡, 𝑤𝑡 
14 


## Page 15

Example: 19-State Random Walk 

In both cases, intermediate values of the bootstrapping parameter (λ or n) 
performed best 

The results with the off-line λ-return algorithm are slightly better at the best 
values of α and λ, and at high α 


## Page 16

Forward (Theoretical) View of λ-return 
Methods 

The forward view defines each update using future rewards and states following the updated 
state 

Intuition: ride along the state sequence, updating each state once from its own vantage point 

After updating a state from its vantage point, the algorithm never revisits that state again 

Future states appear in many updates, each time viewed from an earlier preceding state 

The forward view is theoretical; equivalent, more efficient implementations exist 
16 


## Page 17

Eligibility Traces 
1. The λ-return 
2. TD(λ) 
3. n-step Truncated λ-return Methods 
4. Sarsa(λ) 
 
 
17 


## Page 18

Forward and Backward Views 

In the forward view, the value of a state is updated using one or multiple future 
time steps 

It is mathematically elegant but inconvenient for online, step-by-step learning 

At time t, the agent does not yet observe the rewards that are many steps in the 
future 

Waiting until episode termination to update all visited states wastes data and 
computation 

The backward view replaces “look forward from past states” with local TD 
errors and credit assignment backward in time 

Example of a credit assignment problem: did the bell or the light cause the shock? 
 

Forward and backward views can produce identical or nearly identical learning 
updates 

Eligibility traces (in the backward view) allow more efficient implementations 
18 


## Page 19

The Backward View 

At each time step t, we compute a TD error δt from the most recent transition 

We propagate this TD error backward to earlier states with a decaying influence 
over time 

Picture a TD error δt shouted backward through time, increasingly muffled for 
older states 

The backward view implements λ-returns approximately using only online 
information and local updates 
19 


## Page 20

Eligibility Traces with Function 
Approximation 
With function approximation, we use a weight vector 𝑤𝑡∈ℝ𝑑 
The eligibility trace is another vector 𝑧𝑡∈ℝ𝑑 
The weight vector acts as long-term memory, which accumulates 
knowledge over the whole learning process 
The eligibility trace acts as short-term memory, typically shorter than 
the episode duration 
The trace only affects learning indirectly through its influence on 
later weight updates 
20 


## Page 21

TD(λ) 
TD(λ) is an early, widely used RL algorithm, the first algorithm with 
a formal forward-backward equivalence using eligibility traces 
It empirically approximates the off-line λ-return algorithm 
It improves over off-line λ-return by updating weights on every time 
step 
It also applies naturally to continuing tasks 
21 


## Page 22

Eligibility Traces in TD(λ) 

TD(λ) maintains an eligibility trace for each component of the weight 
vector wt 
The trace vector zt has the same dimension as the weight vector wt  

When a feature is active at time t, the corresponding component of zt is 
increased 

At every time step, all eligibility components decay by a factor γ · λ 

The eligibility trace captures which parameters recently influenced the 
current value estimate and deserve more credit or blame 

Eligibility traces are short-term memory variables that decay over time 
and control how strongly current TD errors update the learned values 
Tabular case: one trace per state (or state-action pair) 
Function approximation: one trace per parameter (or feature) 
22 


## Page 23

Update Equations 
The eligibility trace update: 
  
𝐳𝑡= 𝛾 𝜆 𝐳𝑡−1 + ∇𝑣 𝑆𝑡, 𝐰𝑡 
The term ∇𝑣 𝑆𝑡, 𝐰𝑡 marks the weights that helped produce the 
current estimate as eligible 
The factor γλ ensures that older contributions gradually fade from 
the eligibility trace 
23 
accumulating eligibility trace 
times of visits to a state 


## Page 24

Update Equations 
One-step TD error: 
  
𝛿𝑡= 𝑅𝑡+1 + 𝛾 𝑣 𝑆𝑡+1, 𝐰𝑡−𝑣 𝑆𝑡, 𝐰𝑡 
Weight update with eligibility traces: 
  
𝐰𝑡+1 = 𝐰𝑡+ 𝛼 𝛿𝑡 𝐳𝑡 
vs.  𝐰𝑡+1 = 𝐰𝑡+ 𝛼 𝛿𝑡 ∇𝑣 𝑆𝑡, 𝐰𝑡 
24 


## Page 25

TD(λ) Behavior 
TD(λ) closely matches the ideal offline λ-return algorithm when step 
size α is small enough 
Performance is often best at an intermediate λ value 
Extreme values can decrease efficiency or increase bias 
 
Metaphor: each visited state or feature “glows” after visitation 
The glow intensity fades at rate γλ as time passes 
The current reward prediction error distributes onto all glowing 
states in proportion to their remaining brightness 
25 


## Page 26

Forward-Backward Equivalence 
Each time step, we update the trace zt by adding current features and 
decaying previous entries 
We compute the current TD error δt from the observed reward and 
the bootstrapped next-state value 
We update all weights in proportion to δt and their current eligibility 
zt 
If w is held fixed during an episode, the backward-view TD(λ) 
matches the forward λ-return updates 
The total parameter change then equals that from using forward-
view λ-returns at each time step of the episode 
26 


## Page 27

λ as Temporal Credit-Assignment Memory 
In the backward view, λ controls how long past states remain eligible 
λ = 0: the traces reset immediately; only the current state has 
nonzero eligibility, i.e., TD(0) 
0 < λ < 1: eligibility decays geometrically 
A state k steps back has a trace (γλ)k 
Such states still receive credit from current updates, but less than more 
recent states 
λ = 1: the traces decay only with γ; in episodic tasks every state 
shares credit equally, i.e., TD(1), Monte Carlo 
27 


## Page 28

Eligibility Traces 
1. The λ-return 
2. TD(λ) 
3. Sarsa(λ) 
 
 
28 


## Page 29

From State Values to Action Values: 
Sarsa(λ) 
Control usually needs action values q(s, a), not only state values v(s) 
We approximate action values by 𝑞 𝑠, 𝑎, 𝑤 
Eligibility trace ideas from TD(λ) transfer almost directly to action 
values 
The forward view uses action-value λ-returns; the backward view 
gives Sarsa(λ) 
29 


## Page 30

Action-Value n-step Returns 
Action-value n-step return for 𝑡+ 𝑛 <  𝑇: 
  
𝐺𝑡:𝑡+𝑛= 𝑅𝑡+1 + ⋯+ 𝛾𝑛−1𝑅𝑡+𝑛+ 𝛾𝑛𝑞 𝑆𝑡+𝑛, 𝐴𝑡+𝑛, 𝐰𝑡+𝑛−1  
For 𝑡+ 𝑛 ≥𝑇, 𝐺𝑡:𝑡+𝑛= 𝐺𝑡 
These returns bootstrap from later action-value estimates 
They provide the building blocks for the action-value λ-return 
30 


## Page 31

Forward View λ-return for Action Values 
Combine action-value n-step returns into a λ-return 𝐺𝑡
𝜆 
Definition matches the state-value λ-return, now using 𝐺𝑡:𝑡+𝑛 for 
action values 
Off-line action-value λ-return algorithm: 
  
𝐰𝑡+1 = 𝐰𝑡+ 𝛼𝐺𝑡
𝜆−𝑞 𝑆𝑡, 𝐴𝑡, 𝐰𝑡
∇𝑞 𝑆𝑡, 𝐴𝑡, 𝐰𝑡, 𝑡= 0, … , 𝑇−1 
and 𝐺𝑡
𝜆= 𝐺𝑡:∞
𝜆 in long episodic or continuing tasks 
31 


## Page 32

Sarsa(λ) Update Rule 
Sarsa(λ) approximates the action-value λ-return algorithm by TD 
learning 
Parameter update keeps the TD(λ) form: 
  
𝐰𝑡+1 = 𝐰𝑡+ 𝛼 𝛿𝑡 𝐳𝑡 
Action-value TD error: 
  
𝛿𝑡= 𝑅𝑡+1 + 𝛾𝑞 𝑆𝑡+1, 𝐴𝑡+1, 𝐰𝑡−𝑞 𝑆𝑡, 𝐴𝑡, 𝐰𝑡 
We replace 𝑣 𝑆𝑡, 𝐰𝑡 with 𝑞 𝑆𝑡, 𝐴𝑡, 𝐰𝑡 in all TD(λ) formulas 
32 


## Page 33

Eligibility Traces for Action Values 
Sarsa(λ) maintains an eligibility trace vector zt over parameters 
The trace is initialized at the start of an episode: 𝐳−1 = 0 
For 0 ≤𝑡 ≤𝑇: 
𝐳𝑡= 𝛾 𝜆 𝐳𝑡−1 + ∇𝑞 𝑆𝑡, 𝐴𝑡, 𝐰𝑡 
Some optimizations are possible in the special case of binary features 
33 


## Page 35

Example: Gridworld 

Initial action values are 0, a positive reward is received only at goal state G 

One-step Sarsa increases only the value of the final action in the episode 

n-step Sarsa increases equally the last n action-values when γ = 1 

Sarsa(λ) with λ = 0.9 updates all actions along the path, fading with recency 

Fading credit suits tasks where early actions matter, but are less trusted than later actions 
35 


## Page 36

Conclusions 
λ-return is a geometric average of n-step returns, which interpolates 
between TD(0) and MC bias-variance extremes 
The forward view offers analytical clarity, while the backward view 
with eligibility traces yields equivalent online updates from local TD 
errors 
TD(λ) and Sarsa(λ) share fading-memory credit assignment 
controlled by γλ and step-size α 
Eligibility traces often improve data efficiency dramatically in control 
problems 
36 


## Page 37

Main Reference 
Sutton, R.S. and Barto, A.G. (2018). Reinforcement Learning: An 
Introduction. 2nd edition. MIT Press, Cambridge, MA. 
http://incompleteideas.net/book/the-book-2nd.html 
37 
