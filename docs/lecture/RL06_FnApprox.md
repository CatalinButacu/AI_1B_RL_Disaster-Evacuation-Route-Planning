# RL06_FnApprox

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL06_FnApprox.pdf

**Pages:** 95

---


## Page 1

Reinforcement Learning 
6. Function Approximation 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
2 


## Page 3

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
3 


## Page 4

Memory and Generalization 
So far, we have considered the tabular case 
Memory requirements are high when the number of states is large 
It is impossible to reuse information about one state for non-
neighboring states 
Instead, we want good approximate solutions that require only 
limited computational resources 
We need to generalize from previous encounters with states that are, 
in some sense, similar to the current one 
We can obtain such generalization with function approximation, 
often based on supervised learning 
4 


## Page 5

Example: Pacman 
5 


## Page 6

Large Scale RL 
Reinforcement learning has been used to solve large problems 
Backgammon: 1020 states 
Go: 10170 states 
Helicopter flight: continuous state space 
 
6 


## Page 7

Value Function Approximation 
Value function approximation (VFA) replaces the table with a 
general, usually parameterized, form 
7 


## Page 8

Value Function Approximation 

When we update the parameters, the value estimates of many states 
change simultaneously 

Typically, the number of parameters is much smaller than the number of 
states 

Changing one parameter changes the estimated values of many states 
(generalization) 

Generalization makes RL more powerful but also more difficult to 
manage and understand 

Extending RL with function approximation makes it applicable to 
partially observable problems, where the full state is not available 

In some models (including linear models and neural networks), the 
parameters are weights. Both θ and w are commonly used as notations 
8 


## Page 9

Function Approximation in RL 
Many supervised methods can approximate functions, such as linear 
models, neural networks or decision trees 
RL needs online methods that learn from data that arrives 
incrementally during interaction 
RL requires function approximators that handle changing target 
functions over time 
Methods that rely on static datasets or fixed targets are usually 
unsuitable for RL 
In RL, the type of supervised function approximation is regression  
(state → target value), not classification 
9 


## Page 10

The Objective Function 
In the tabular case, a continuous measure of prediction quality is not 
necessary because the learned value function can become equal to the 
true one, and updates affect only single states 
With function approximation, updating one state value affects many 
others 
More states than weights implies trade-offs: improving one state 
worsens others 
A state distribution μ(s) specifies how much we care about error in 
each state (μ(s) is the normalized number of visits to s) 
A natural objective function is the mean squared error between 
values – the mean squared value error: 
10 


## Page 11

Gradient Methods 
Minimizing VE requires selecting a function approximator and 
optimization strategy suited to the RL context 
Linear gradient methods allow analysis and often guarantee 
convergence to the global VE minimum 
Nonlinear methods lack convergence guarantees and may require 
careful tuning or constraints 
11 


## Page 12

GD vs. SGD 
Gradient descent (GD) assumes access to all states or full training 
data for each update 
It computes an exact objective gradient, then takes one step in the 
true direction 
Stochastic gradient descent (SGD) updates from single samples or 
mini-batches, not the whole distribution 
Each stochastic update uses a noisy gradient estimate that 
approximates the true gradient 
RL agents interact online and data arrives sequentially, thus SGD is a 
natural choice here 
SGD approximates GD over time 
12 


## Page 13

SGD in RL 
13 


## Page 14

SGD in RL 
14 


## Page 15

The Target 
15 


## Page 16

16 


## Page 17

Gradient Monte Carlo Algorithm 
Each episode provides full return Gt​ as an unbiased training target 
for each visited state 
Update rule: 
 
 
Suitable for episodic tasks where full returns can be observed and 
stored 
High variance of returns makes MC slower than bootstrapping 
methods in some cases 
 
[
(
,
)]
(
,
)
t
t
w
t
w
w
G
v S w
v S w





17 


## Page 18

True value           is often unknown; we use sample-based targets Ut 
Monte Carlo target Ut = Gt​ is unbiased: 
Bootstrapped targets, e.g.,                              depend on current 
weights and are biased 
Substituting such biased targets breaks the true gradient descent 
nature of the update 
These updates are called semi-gradient methods 
 
Semi-Gradient Methods (for Bootstrapping) 
[
|
]
( )
t
t
G
S
s
v
s



1
1
(
,
)
t
t
t
R
v S
w




( )
v
s

18 


## Page 19

19 


## Page 20

Convergence 
20 


## Page 21

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
21 


## Page 22

Linear Function Approximation 
22 


## Page 23

Gradients for Linear Models 
23 


## Page 24

Convergence 
24 


## Page 25

Convergence 
Linear function approximators have a unique global optimum or a 
flat set of equivalent optima 
Gradient Monte Carlo converges to the global minimum of VE if the 
step size α decreases appropriately 
Semi-gradient TD(0) converges under linear function approximation, 
but not to the global minimum of VE 
TD(0) converges to a fixed point close to the global minimum 
   
𝑉𝐸(𝐰𝑇𝐷) ≤
1
1 −𝛾min
𝐰𝑉𝐸(𝐰) 
  
Often γ is close to 1, but TD methods still work well in practice 
25 


## Page 26

Example 
MDP: 
Linear model: 
26 


## Page 27

Program 
1 Simple Grid → LinearTDAgent.py 
with 2 policies, default reward –0.04 
 
Policy 1  
up, up, right, right, right, then random 
If the number of steps exceeds 20, terminate with 0 
Learned weight vector w for policy 1: [-0.306, 0.127, 0.275] 
 
Approximate value function over the grid for policy 1: 
0.65    0.78    0.90    [1] 
0.37    [###]   0.63    [-1] 
0.10    0.22    0.35    0.48 
27 


## Page 28

Program 
1 Simple Grid → LinearTDAgent.py 
with 2 policies, default reward –0.04 
 
Policy 2  
right, right, up, up, right, then random 
If the number of steps exceeds 20, terminate with 0 
Learned weight vector w for policy 2: [-0.605, -0.038, 0.148] 
 
Approximate value function over the grid for policy 2: 
-0.20   -0.24   -0.28   [1] 
-0.35   [###]   -0.42   [-1] 
-0.50   -0.53   -0.57   -0.61 
28 


## Page 29

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
29 


## Page 30

Feature Construction 
Especially with simple approximation models, like linear ones, direct 
state encoding may not offer enough information for problem 
representation and generalization 
Constructed features can turn a simple linear model into a nonlinear 
approximator over the original state space 
Constructed features may include domain knowledge 
A state can be represented by a feature vector, e.g.: 
The distance of a robot from some landmarks 
Piece configurations in backgammon 
 
30 


## Page 31

Polynomials 
Some problems have states with numeric dimensions such as s1 and 
s2. Using only (s1, s2) ignores interactions and yields value 0 when 
both are 0 
A feature vector like (1, s1, s2, s1 · s2) adds an intercept and another 
term that can help capture interactions between dimensions 
Adding more polynomial features approximates complex interactions 
while the model remains linear in weights 
Polynomial models can be useful for linear model approximation, but 
less useful for neural networks 
31 


## Page 32

Example: Pole Balancing 
In the pole balancing task, high angular velocity can be either good or 
bad depending on the angle 
A linear value function cannot represent this if these features are 
coded separately for the angle and the angular velocity 
32 


## Page 33

Example: Mountain Car 
33 


## Page 34

Program: 2 Feature Construction → 1 Poly.py 


## Page 35

Coarse Coding 
Coarse coding uses overlapping features 
Each binary feature is active if the state falls inside a predefined 
receptive field 
A state is represented by the set of features whose regions contain it 
Feature overlap leads to generalization across states, based on shared 
active features 
Binary features simplify representation: active features are 1, inactive 
features are 0 
35 


## Page 36

Example 

Generalization from state s to state s' depends on the number of their 
features whose receptive fields (in this case, circles) overlap 

These states have one feature in common, so there will be slight 
generalization between them 
36 


## Page 37

Generalization in Coarse Coding 
Small receptive fields cause narrow generalization 
Large fields produce broader generalization 
Feature shape and orientation (e.g., circles, ellipses) influence the 
pattern and direction of generalization 
Carefully selected receptive fields allow precise control over locality 
and extent of generalization 
37 


## Page 38

Form Impacts Generalization 
38 


## Page 39

Example 
39 
explanations in the next slide 


## Page 40

Example 

An example of the strong effect of a feature width on initial generalization 
(first row) and weak effect on asymptotic accuracy (last row)  

A 1D square wave function is learned using linear function approximation 
with coarse coding 

States are represented by overlapping 1D interval features of three widths 

All 3 setups use about 50 features across the range and randomly sampled 
training examples 

Broad receptive fields give wide generalization and smooth updates over 
many states early 

Narrow receptive fields change only nearby states, so the learned function 
looks bumpier initially 

Eventually, all receptive field widths yield similar final approximations 

Shape mainly affects generalization behavior 
40 


## Page 41

Program: 2 Feature Construction → 2 Coarse.py 
41 


## Page 42

Tile Coding 
Tile coding uses multiple tilings of the state space, each partitioned 
into non-overlapping tiles 
It may be the most practical feature representation  
A state activates one tile per tiling, resulting in a sparse binary 
feature vector 
Multiple tilings with offsets allow overlapping receptive fields for 
generalization 
With n tilings, exactly n features are active per state, regardless of 
state location 
Each tile corresponds to one component of the weight vector used in 
value approximation 
42 


## Page 43

Example 
43 
The feature vector x(s) has one component for each tile in each tiling. 
Here, there are 4 × 4 × 4 = 64 components, all of which will be 0 except for the four 
corresponding to the tiles that s falls within. 
 


## Page 44

Advantages of Tile Coding 
Feature representation is consistent and enables uniform learning 
The overall number of features that are active at one time is the same for any 
state 
Exactly one feature is present in each tiling, so the total number of features 
present is always equal to the number of tilings 
Tile coding supports high resolution learning with manageable 
computational cost 
Sparse binary features allow fast computation and efficient updates via index 
lookups 
Learning rate can be scaled with tiling count: step size α = 1 / n​ yields 
exact one-step learning, i.e., 𝑣 (𝑆𝑡, 𝑤𝑡) becomes target Ut in one step 
Slower rates are possible, e.g., α = 1 / 10n 
 
 
44 


## Page 45

Uniform Offsets 
45 


## Page 46

Asymmetrical Offsets 

Asymmetrical offsets are preferred in tile coding 

If the tilings are uniformly offset, then there are diagonal artifacts and 
substantial variations in the generalization 

With asymmetrically offset tilings, the generalization is more spherical and 
homogeneous 


## Page 47

Offset Recommendation 
For a continuous space of dimension k, a good choice is to use the 
first odd integers (1, 3, 5, 7, . . . , 2k – 1), with the number of tilings n 
set to an integer power of 2 greater than or equal to 4k 
In the previous figure: k = 2, n = 23 ≥ 4k, and displacement vector  
(1, 3) 
In 3D, the first four tilings would be offset in total from a base 
position by (0, 0, 0), (1, 3, 5), (2, 6, 10), and (3, 9, 15) 
47 


## Page 48

Program: 2 Feature Construction → 3 Tile.py 


## Page 49

Example: Random Walk 

The space of 1000 states is treated as a single continuous dimension, 
covered with tiles each 200 states wide 

The multiple tilings are offset from each other by 4 states 

The step-size parameter is set so that the initial learning rate in the two 
cases is the same, α = 0.0001 for the single tiling and α = 0.0001 / 50 for the 
50 tilings 
Learning curves on the 1000-state 
random walk example for the gradient 
MC algorithm with a single tiling and 
with multiple tilings 


## Page 50

Radial Basis Functions (RBFs) 

RBF features vary continuously from 1 at center to 0 as distance increases 
(Gaussian shape) 

RBF networks produce smooth, differentiable approximations over 
continuous input spaces. 

RBFs support local generalization tuned by width σ 
50 
1D RBF 


## Page 51

Discussion 
RBFs produce approximate functions that vary smoothly and are 
differentiable 
However, in most cases this has no practical significance  
The computational cost of RBFs is higher due to exponentials 
Tile coding has better performance in high-dimensional cases 
51 


## Page 52

Program: 2 Feature Construction → 4 Rbf.py 
52 


## Page 53

Generalization Structure 
Tile/coarse coding 
Local overlapping regions ⇒ controlled, local generalization 
A weight update at one state affects nearby states that share active tiles 
RBFs 
Similar locality but smooth, because features decay with distance 
Polynomials 
Global features 
A change to one weight affects the entire space 
53 


## Page 54

Inductive Bias (Prior Knowledge) 
The approximator is just a mechanism; the encoding is where most 
of the prior knowledge is contained in classic RL 
Assumptions: 
Nearby states should have similar values → local features (tiles, RBFs) 
The value surface is smooth and low-curvature → low-degree polynomials 
The function might have sharp discontinuities → more tiles, finer grids 
54 


## Page 55

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
55 


## Page 56

Nonlinear Value Function Approximation 

There exist several nonlinear methods for 
approximating the value function, such as: 

Neural Networks (NNs) 

Memory-based (nonparametric) functions 

NNs have recently become the most 
popular approximators 

NNs are universal function approximators 

In deep architectures they can generate 
hierarchical representations of features 
automatically (vs. hand crafted features) 

They typically learn by stochastic gradient 
methods 
56 


## Page 57

NNs for TD Learning 
NNs can be trained with TD errors to estimate value functions 
The update rule adjusts the weights to reduce TD error 
NN gradient computation applies equally in supervised and 
reinforcement learning settings 
Function approximation with NNs allows generalization between 
states or state-action pairs 
57 


## Page 58

Challenges with Deep Networks 
Deeper networks can overfit due to high capacity and limited data; 
regularization is often needed 
Gradients may vanish during backpropagation, impairing learning in 
early layers 
Generalization performance may degrade when adding more layers 
despite increased expressivity 
In online RL, overfitting is less critical, but generalization between 
trajectories still matters 
58 


## Page 59

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
59 


## Page 60

Instance-Based Models 
Learning based on similarity 
Predict the value of an instance using those of similar instances 
Nearest neighbors 
1NN: return the value of the most similar instance 
kNN: average over the k nearest neighbors, usually with a weighting scheme 
(e.g., w = 1 / d) 
Key issue: the distance metric, e.g., Euclidean distance 
Trade-offs: small k gives relevant neighbors, large k gives smoother, more 
global functions 
60 


## Page 61

Non-parametric Models 
Parametric models: 
Fixed set of parameters 
More data means better settings 
Non-parametric models: 
Complexity of the classifier increases with data 
Better in the limit, often worse in the non-limit 
kNN is a non-parametric method 
Usually performance decreases for high-dimensional problems 
The concept of distance becomes less relevant in high dimensions 
 
61 


## Page 62

kNN 
62 


## Page 63

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
63 


## Page 64

Control with Approximation 
64 


## Page 65

Episodic Semi-Gradient Control 
65 


## Page 66

Sarsa 
66 


## Page 67

67 


## Page 68

Example: Mountain Car Problem 
Mountain Car presents a standard continuous control benchmark 
State includes car position and velocity on a one-dimensional track 
Actions: full throttle left, zero throttle, or full throttle right 
Reward is –1 each time step until reaching the goal 
The optimal behavior requires reversing to gain momentum before 
climbing 
68 


## Page 69

Implementation Details 
69 


## Page 70

Learning 
Cost-to-go 
increases 
70 
є = 0 


## Page 71

Learning 
Low (good) cost-to-go appears near the goal region at the hilltop 
High (bad) cost-to-go appears in regions where the car remains stuck 
The agent learns a policy to back up left, then accelerate right to 
reach the goal 
71 


## Page 72

Learning 
72 


## Page 73

Program 
3 Control - Sarsa → Sarsa_Agent.py, MountainCar_Env.py 
73 


## Page 74

74 


## Page 75

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
75 


## Page 76

Continuing Tasks and Average Reward 
Many tasks in reinforcement learning run indefinitely; there is no 
natural episode limit or reset 
For example, an elevator controller runs indefinitely. We want to 
minimize the average waiting time in the long term, so favoring 
immediate rewards would be arbitrary 
Short-term and delayed rewards have equal importance; the discount 
factor γ is no longer important 
We define the goal of a policy π as its average reward r(π) per step 
76 


## Page 77

Definition of Average Reward 
The average reward r(π) is the time-average of expected rewards 
under policy π 
𝑟𝜋= lim
ℎ→∞
1
ℎ 𝔼𝑅𝑡
𝜋
ℎ
𝑡=1
 
This quantity measures reward per time step, not total reward over a 
finite horizon 
Policies are compared and considered optimal according to their 
average rewards r(π) 
77 


## Page 78

Steady-State and Ergodicity 
The fraction of time spent in each state then depends only on the 
policy and dynamics 
We call this steady long-run distribution over states the steady-state 
distribution 
When this distribution exists and is unique, the Markov decision 
process is called ergodic 
An ergodic system forgets where it started; state frequencies stabilize 
over time 
In that case, the average reward r(π) is well defined and independent 
of the start 
78 


## Page 79

Differential Return 
Raw cumulative reward often diverges in continuing tasks, so we 
measure reward relative to the average 
We define the differential return Gt by subtracting r(π) from each 
future reward: 
𝐺𝑡= 𝑅𝑡+1 −𝑟𝜋+ 𝑅𝑡+2 −𝑟𝜋+ ⋯ 
A positive Gt means this trajectory segment performed better than 
the usual average of the policy π 
A negative Gt means that performance was worse than average, 
under the same policy π 
79 


## Page 80

Differential Value Functions 
The differential state value 𝑣𝜋𝑠 is the expected differential return 
from state s 
The differential action value 𝑞𝜋𝑠, 𝑎 is the expected differential 
return from (s, a) using policy π: 
𝑞𝜋𝑠, 𝑎= 𝔼𝜋𝐺𝑡
𝑆𝑡= 𝑠, 𝐴𝑡= 𝑎 
These values measure how much better or worse than average a 
state or action is 
If we add the same constant to every differential value, choices and 
rankings remain unchanged 
80 


## Page 81

Bellman Equation for Differential q-Values 
Differential action values satisfy a Bellman relation similar to the 
discounted case, without γ 
For a fixed policy π, 𝑞𝜋𝑠, 𝑎 equals a centered immediate reward 
plus an expected next 𝑞𝜋: 
𝑞𝜋𝑠, 𝑎=  𝑝𝑠′, 𝑟
𝑠, 𝑎
𝑟−𝑟𝜋+  𝜋𝑎′
𝑠′ 𝑞𝜋𝑠′, 𝑎′
𝑎′
𝑠′,𝑟
 
The term r – r(π) measures how good the immediate reward is 
relative to average 
The second term propagates the expected differential value of the 
next state-action pair 
81 


## Page 82

TD Learning with an Average-Reward 
Baseline 
The algorithms keep an estimate 𝑅𝑡 of the average reward r(π) while 
learning 

𝑅𝑡 is an estimate at time t of the average reward r(π) 
Each step produces a differential TD error : 
𝛿𝑡= 𝑅𝑡+1 −𝑅𝑡+ 𝑞 𝑆𝑡+1, 𝐴𝑡+1, 𝑤𝑡−𝑞 𝑆𝑡, 𝐴𝑡, 𝑤𝑡 
 
82 


## Page 83

Differential Semi-Gradient Sarsa 
We approximate action values with a differentiable function 
𝑞 𝑠, 𝑎, 𝐰 with weights w 
A behavior policy, usually є-greedy, selects actions based on current 
estimates 𝑞  
After each transition, we update the average reward and the weights 
based on the TD error δt: 
  
𝑅 ←𝑅 + 𝛽 𝛿 
  
𝐰←𝐰+ 𝛼 𝛿 ∇𝐰𝑞 𝑆, 𝐴, 𝐰 
 
 
83 


## Page 84

84 


## Page 85

Example: Access-Control Queuing 
Environment: 10 servers and a single queue of customers with 4 
priority levels 
Serving a customer yields rewards 1, 2, 4, or 8 depending on priority; 
rejecting yields 0 reward 
The queue never empties; servers become free with probability  
p = 0.06 each time step 
The agent chooses accept or reject from states described by the 
number of free servers and the priority of the customer at the head 
of the queue 
Goal: maximize long-run reward, so average reward is appropriate 
85 


## Page 86

Results 
Learning uses differential semi-gradient one-step Sarsa with  
α = 0.01, β = 0.01, є = 0.1 
Training for 2 million steps 
Estimated long-run average reward 𝑅  converges to ~ 2.31 units per 
time step 
The learned policy accepts higher-priority customers more often and 
only accepts lower-priority customers when more servers are free 
86 


## Page 87

The drop on the right of the graph is probably 
due to insufficient data; many of these states 
were never experienced 


## Page 88

Discounted Setting vs. Average Reward  
in Continuing Tasks 
With function approximation, many distinct states can share the 
same features, so the agent cannot treat them differently 
In the extreme, its behavior depends only on long-run reward 
statistics of the process, not on individual state identities 
If we average undiscounted rewards over time, we get the average 
reward r(π) for the policy 
If we average discounted returns over time, we get 
1
1 −𝛾𝑟𝜋 
The policies rank exactly the same 
88 


## Page 89

Role of γ in Continuing Tasks with Function 
Approximation 
In theory, maximizing the discounted value over on-policy states is 
equivalent to maximizing the average reward 
γ does not change which policy is optimal 
It does not define the control problem 
Control algorithms with function approximation do not truly 
optimize either discounted value or average reward 
The policy improvement theorem fails, so no variant guarantees 
reliable policy improvement in practice 
In continuing tasks, the average reward defines the objective (what) 
The discount factor γ mainly controls how learning behaves 
Bias-variance trade-offs, learning stability 
 
89 


## Page 90

Function Approximation 
1. On-Policy Prediction with Approximation 
 
1.1. Value-Function Approximation  
 
1.2. Stochastic-Gradient and Semi-Gradient Methods 
 
1.3. Approximation with Linear Methods  
 
1.4. Feature Construction (Polynomials, Coarse Coding, Tile Coding, RBF) 
 
1.5. Approximation with Neural Networks  
 
1.6. Approximation with Memory-Based Methods 
2. On-Policy Control with Approximation  
 
2.1. Episodic Semi-Gradient Control  
 
2.2. Average Reward for Continuing Tasks  
 
2.3. The Deadly Triad  
 
 
90 


## Page 91

The Deadly Triad 
Instability and divergence in RL arise when three elements appear 
together: the deadly triad 
Function approximation uses parametric value models instead of large 
tables 
Bootstrapping uses targets that include current value estimates 
Off-policy learning learns about one policy while following another behavior 
policy 
Each ingredient is useful alone but dangerous in combination 
91 


## Page 92

Why the Combination Can Diverge 

All three together can cause value estimates to diverge even in simple 
prediction tasks 

Function approximation couples many states through shared parameters 
and updates 

Bootstrapping creates feedback by providing estimates into update 
targets 

Off-policy learning updates under a distribution that does not match that 
of the target policy 

The deadly triad definition applies to any function approximator, not 
only linear models 

Usually, function approximation and bootstrapping are important (large 
problems, continuing tasks), but so is off-policy learning in some cases 
92 


## Page 93

Off-Policy Learning 
Predictive knowledge view uses many value functions for many tasks 
An agent can follow one behavior policy while learning about many 
target policies 
A robot in a building follows one safe behavior policy (wander safely, avoid 
collisions) yet wants to learn many “what if” predictions at once: what if I 
followed the wall, what if I walked fast, what if I went to the charger now 
On-policy learning alone cannot cover many hypothetical behaviors 
efficiently 
Off-policy TD reuses a single experience stream for multiple 
predictive questions 
The deadly triad highlights instability risks in large-scale off-policy 
systems 
93 


## Page 94

Conclusions 
Function approximation enables RL in large or continuous state 
spaces with limited memory resources 
Value function approximation uses stochastic gradient and semi-
gradient methods to learn estimates from interaction 
Feature construction determines representations and thus 
generalization patterns, which affect stability, sample efficiency and 
performance 
Neural networks offer powerful nonlinear approximators but 
introduce optimization difficulties, instability risks and overfitting 
concerns 
The deadly triad is the combination of function approximation, 
bootstrapping and off-policy learning, which often leads to instability 
 
94 


## Page 95

Main References 
Sutton, R. S. and Barto, A. G. (2018). Reinforcement Learning: 
An Introduction. 2nd edition. MIT Press, Cambridge, MA. 
http://incompleteideas.net/book/the-book-2nd.html 
 

Castellini, A. (2023). On-Policy Prediction with Approximation, 
Reinforcement learning – LM Artificial Intelligence, University of Verona, 
https://profs.scienze. univr.it/~castellini/docs/reinforcementLearning22-
23/RL_L9_OnPredApprox.pdf 

Castellini, A. (2023). On-Policy Control with Approximation and Deep Q 
Networks (DQN), Reinforcement learning – LM Artificial Intelligence, 
University of Verona, https://profs.scienze.univr.it/~castellini/docs/ 
reinforcementLearning22-23/RL_L10_OnControlApprox.pdf 
 
95 
