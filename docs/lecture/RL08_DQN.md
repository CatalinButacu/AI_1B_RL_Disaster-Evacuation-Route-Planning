# RL08_DQN

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL08_DQN.pdf

**Pages:** 76

---


## Page 1

Reinforcement Learning 
8. Deep Q-Networks 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

Deep Q-Networks 
1. Standard DQN 
2. Double DQN 
3. Dueling DQN 
4. Rainbow DQN 
2 


## Page 3

Short Timeline 
Standard DQN: 2015 
Double DQN: 2016 
Prioritized Experience Replay: 2016 
Dueling DQN: 2016 
Distributional Q-learning (C51): 2017 
Noisy Networks: 2018 
Rainbow DQN: 2018 


## Page 4

Deep Q-Networks 
1. Standard DQN 
2. Double DQN 
3. Dueling DQN 
4. Rainbow DQN 
4 


## Page 5

NN Approximation of Q Function 
Tabular Q-learning stores one value per state-action pair in a table 
It works for small, discrete state spaces with limited observations 
Rich sensory inputs (images, large vectors, text) create enormous, 
near-continuous state spaces 
We can represent the Q function as 𝑄𝑠, 𝑎; 𝜃 with a neural network 
with parameters 𝜃 
Input: state s; output: Q-values 
The policy still selects the action with the maximum predicted Q-value 
Deep networks can process raw inputs, form hierarchical features, 
and output action values in one model 
Hence the name Deep Q-Network (DQN) (Google DeepMind, 2015) 
5 


## Page 6

Atari 2600 Games 
49 games: Pong, Breakout, Space Invaders, Seaquest, Beam 
Rider, etc. 
 
 
 
 
DQN was intended as a step toward artificial general 
intelligence, because it uses the same algorithm for quite 
different games 
6 


## Page 7

Input Preprocessing 
A human player sees 210 x 160 images with 128 RGB colors 
For DQN, each frame is converted into an 84 x 84 matrix of 
luminance values 
L = 0.2126 · R + 0.7152 · G + 0.0722 · B 
7 


## Page 8

Game Model 
Inputs: 4 consecutive game frames, so the agent can estimate the 
speed and direction of characters or objects in the game 
Outputs: the Q-values of all possible actions 
The number of actions depends on the game and can range from 2 to 
18 (matching joystick commands), e.g., up, down, right, left, fire, 
accelerate, brake, pick up the key, open the door, etc. 
The network interacts with an Atari game emulator 
Convolutional neural networks (CNN) are used 
The environment provides feedback only through images and score 
8 


## Page 9

Q Function Approximation 
Direct approach 
Optimized approach (in DQN): the outputs are 
the Q values of all possible actions in the current 
state. They are all computed in a single step.  
For each game, the number of outputs equals 
the number of valid actions 


## Page 10

Q-Learning Updates 
Tabular Q-Learning update: 
  
 
Consider a transition 𝑠𝑡, 𝑎𝑡, 𝑟𝑡+1, 𝑠𝑡+1  
The one-step TD target in naive deep Q-learning would be: 
  
𝑦𝑡= 𝑟𝑡+1 + 𝛾max
𝑎′ 𝑄𝜃𝑠𝑡+1, 𝑎′  
The network predicts 𝑄𝜃𝑠𝑡, 𝑎𝑡 
The squared TD error for this sample is: 
  
𝐿𝑡𝜃= 𝑦𝑡−𝑄𝜃𝑠𝑡, 𝑎𝑡
2 
This looks similar to supervised learning, but the target 𝑦𝑡 depends 
on the current network 
10 


## Page 11

Moving Target Problem 
After each gradient step, the network changes and therefore the 
target changes as well 
Distorted Q-values can enter the targets and then propagate or even 
amplify over time 
The learning process follows a target that keeps moving as the 
network updates 
 
iterations 
11 


## Page 12

Training Instability 
Problems: 
The target values are not fixed 
Successive experiences are correlated and depend on the policy 
Small changes to the parameters cause large changes in the 
policy, which lead to large shifts in the data distribution 
Solutions: 
Fixed target Q-network 
Experience replay 
Clipped error 
12 


## Page 13

Target Network and Loss 
DQN introduces 2 networks with the same architecture: 

The online network 𝑄𝜃𝑠, 𝑎 interacts with the environment and is updated by 
gradient descent 

The target network 𝑄𝜃−𝑠, 𝑎 is a delayed copy used only to compute TD targets 
For a transition 𝑠, 𝑎, 𝑟, 𝑠′  the DQN target is: 
  
𝑦= 𝑟+ 𝛾max
𝑎′ 𝑄𝜃−𝑠′, 𝑎′  
The loss for the online network is: 
  
𝐿𝜃= 𝑦−𝑄𝜃𝑠, 𝑎
2 
  
or, usually, a minibatch average of this expression 
 
13 


## Page 14

Target Network as a Stabilizer 
The target network stays fixed for a while, so the mapping from 𝑄𝜃 
to targets y changes more slowly 
This slower change breaks the tight positive feedback loop that 
causes runaway updates 
The online network can take many gradient steps toward the current 
target network before the target changes 
The design mimics a teacher-student scenario: 

The target network 𝑄𝜃− acts as a teacher that provides temporary values 

The online network 𝑄𝜃 acts as a student that tries to match those values 
After some training, the student parameters replace the teacher 
parameters 
14 


## Page 15

Hard and Soft Target Updates 
Hard target updates copy the online network parameters every C 
steps: 
𝜃−←𝜃    every  𝐶 updates 
Between the hard updates, the target network stays fixed 
Soft target updates use an exponential moving average: 
  
  
𝜃−←𝜏𝜃+ 1 −𝜏𝜃−   with a small 𝜏∈0,1  
Hard updates give clear separation between the student and the 
teacher 
Soft updates produce a smoother evolution of the target network and 
allow a fine control via 𝜏 
In both cases the target network changes more slowly than the online 
network, which stabilizes learning 
 
15 


## Page 16

Target Update Time Scale 
Very frequent target updates remove the stabilizing separation from 
the online network 
Very rare updates create stale teachers that ignore the latest 
understanding 
Update period or 𝜏 controls how quickly the target tracks 
improvements 
Appropriate lag prevents immediate reuse of each fresh estimate as 
its own target 
This delay reduces the chance of runaway bootstrapping feedback 
16 


## Page 17

Experience Replay 
The agent can update parameters only once per environment step if 
it uses strictly online updates 
Each transition is used for a single gradient step and then discarded 
Valuable or rare experiences cannot influence learning for very long 
if they are not replayed 
Consecutive transitions may be strongly correlated  
If the best action is usually “go right”, the training data will be dominated by 
“go right” 
Sudden changes in the policy or environment can drive the network 
into poor regions of parameter space 
17 


## Page 18

Replay Memory / Buffer 
During gameplay, all transitions (s, a, r, s') are stored in a structure 
called replay memory or replay buffer 
When the network is trained, random minibatches from the replay 
memory are used instead of the most recent transition 
This method avoids the problem that successive training samples are 
too similar, which would push the network toward a local optimum 
It is also possible to collect transitions from a human player’s game 
and train the network on those 
 
18 


## Page 19

Experience Replay 
An action is chosen in an є-greedy way 
The transition (st , at , rt+1 , st+1) is added to the replay memory 
The system moves to state st+1 and the game continues, but the 
network weights are updated using a small number of transitions 
sampled from the replay memory (the current transition may be 
used as well but only from the replay memory) 
19 


## Page 20

Details 
DQN maintains a replay memory D with capacity N 
At each time step the agent observes a transition 𝑠𝑡, 𝑎𝑡, 𝑟𝑡+1, 𝑠𝑡+1  
and stores it in D 
When D is full, the oldest transitions are removed to make room for 
new ones 
The replay buffer aggregates experience from many episodes and 
stages of learning 
However, very old transitions may mislead if the dynamics or visited 
regions have changed, but FIFO replacement limits their influence 
20 


## Page 21

Reward Normalization and Error Clipping 
Atari games have very different raw score scales across tasks 
DQN standardizes rewards to reduce this variation 
r = +1 if the game score increases 
r = –1 if the game score decreases 
r = 0 otherwise 
The TD error is clipped to [–1, 1] 
 
 
These two methods limit the size of parameter updates 
The same parameters can work for many different games 
21 


## Page 22

є-greedy Policy 
DQN typically uses an є-greedy policy for exploration 
With probability є, the agent chooses a random action 
With probability 1 – є, the agent chooses 𝑎𝑡= argmax𝑎𝑄𝜃𝑠𝑡, 𝑎 
At the start of training, є often equals 1 to encourage wide 
exploration 
Over time, є decays linearly or according to a schedule down to a 
small value such as 0.1 
The decay schedule balances exploration of new behaviors and 
exploitation of learned Q-values 
22 


## Page 23

23 
Each episode is a complete game 
t represents each step of the game 
φ represents the processed images x 


## Page 24

Full DQN Architecture 
No pooling because positions are very important during gameplay 
24 


## Page 25

Practical DQN Techniques 
The agent selects a new action only every k frames, typically k = 4 
(frame skipping) 
During skipped frames the environment repeats the last chosen 
action 
Frame skipping reduces computation and still captures relevant 
dynamics 
The network input also includes several recent frames (also 4 in the 
original DQN), which encodes motion and short-term history 
Optimization uses minibatch stochastic gradient descent with 
RMSProp or Adam 
25 


## Page 26

Training Methodology 
DQN learned each game by interacting with the game emulator for 
50 million frames, equivalent to 38 days of gameplay 
To evaluate performance after learning, for each Atari game the 
score was computed as the average over 30 games of 5 minutes each. 
Each game began from a random initial state 
The same hyperparameters and network architecture were applied to 
all games 
26 


## Page 27

Results 

The x axis shows the score 
obtained by the DQN 
model as a percentage 
relative to human players 

In the game Montezuma’s 
Revenge, where DQN gets 
0%, the hero can die very 
quickly and the network 
fails to learn 


## Page 28

Example: Breakout 
28 
https://www.youtube.com/watch?v=V1eYniJ0Rnk 


## Page 29

Strengths and Limitations 
DQN learns directly from raw high-dimensional observations 
(images) 
It uses a single network architecture and learning rule for many tasks 
It demonstrates that general-purpose deep reinforcement learning 
(DRL) can reach human-level performance in some games 
 
It tends to overestimate the Q-values 
It requires many environment interactions and substantial 
computation (its sample efficiency is low) 
29 


## Page 30

Deep Q-Networks 
1. Standard DQN 
2. Double DQN 
3. Dueling DQN 
4. Rainbow DQN 
30 


## Page 31

Max-Induced Bias 
Standard DQN target for transition 𝑠′, 𝑟: 
 
𝑦DQN = 𝑟+ 𝛾max
𝑎′ 𝑄target 𝑠′, 𝑎′  
The max operator assumes the largest estimated value is also the 
most accurate 
Noisy value estimates make this assumption systematically optimistic 
Overestimation enters directly into every bootstrapped target and 
distorts value predictions, which results in suboptimal policy 
learning 
 
 
31 


## Page 32

Example 
Consider 2 actions with true values both equal to 10 
Network estimates fluctuate, e.g. (9, 11) in one visit, and (12, 8) in 
another 
Noise is symmetric around 10 (unbiased) but affects each action 
differently 
max(9, 11) and max(12, 8) are both greater than 10 
The average of the maximum is a biased overestimate of the true 
maximum 
32 


## Page 33

33 


## Page 34

Example (cont.) 
The 2 independent estimators QA and QB are both noisy but unbiased 
around 10 
QA(s', ·) = (9, 11); QB(s', ·) = (12, 8) 
Updating QA: a* = argmaxa QA(s', a) = a2 
The target for QA uses the other network: QB(s', a*) = 8, not  
maxa QA = 11 
Next update may flip roles, so sometimes estimates fall above 10, 
sometimes below 
Since selection and evaluation use independent noise, these errors 
cancel, and the expected value is 10 
 
34 


## Page 35

Double DQN 
The online network 𝑄𝑠′, 𝑎′; 𝜃 selects the best action via 
arg max
𝑎′ 𝑄𝑠′, 𝑎′; 𝜃 
The target network 𝑄𝑠′, 𝑎′; 𝜃′  evaluates the value of that selected 
action 
The target is: 
35 
Here we use the 𝜃′ notation for the target network instead of 𝜃−,  
but it is exactly the same concept 


## Page 36

Details: Action Selection 
First, we compute the Q values of all the next state-action pairs using 
the online (main) network θ, and we select action a', which has the 
maximum Q value:  
 
36 


## Page 37

Details: Q Value Computation 
Once we have selected action a', we compute the Q value using the 
target network 𝜃′ for the selected action a' 
 
37 


## Page 38

DQN vs. Double DQN 
38 


## Page 40

Results 
Double DQN outperformed standard DQN in most metrics for the 
same Atari games 
Learning curves typically look smoother and more stable across 
games 
Many scores improve substantially, e.g., Road Runner and Double 
Dunk 
In Wizard of Wor and Asterix, DQN values can grow by orders of 
magnitude 
The actual performance may stagnate or even degrade during this 
explosion 
Double DQN keeps value growth modest and aligned with score 
improvements 
40 


## Page 41

Results: Human-Normalized Scores  
Game 
DQN 
Double DQN 
Wizard of Wor 
67.49 % 
110.67 % 
Asterix 
69.96 % 
180.15 % 
Road Runner 
232.91 % 
617.42 % 
Double Dunk 
17.10 % 
396.77 % 
41 


## Page 42

Example: Road Runner 
42 


## Page 43

Deep Q-Networks 
1. Standard DQN 
2. Double DQN 
3. Dueling DQN 
4. Rainbow DQN 
43 


## Page 44

Motivation 
In many states, actions have similar outcomes; standard Q-networks 
model all (state, action) pairs equally, but this is inefficient 
The dueling network learns common state information, and only 
computes specific action information when necessary 
This improves generalization, especially in environments with 
redundant or uninformative actions 
 
44 


## Page 45

The Advantage Function 
Q(s, a) gives the expected return for action a in state s 

𝑉𝑠= 𝔼𝑎∼𝜋𝑄𝑠, 𝑎 estimates the value of a state under policy π 
The advantage function A(s, a) = Q(s, a) – V(s) measures how much 
better an action is compared to the average 
A large positive advantage indicates an especially good action in that 
state 
45 


## Page 46

Dueling DQN 
Dueling DQN introduces a network that separately estimates state 
value and action advantage 
The network shares initial layers, then splits into two streams: one 
for V(s), one for A(s, a) 
This architecture accelerates learning by modeling only meaningful 
action differences when necessary 
 
In a sense, V and A compete (duel) to explain Q 
Is the state good, or is a particular action better? 
 
46 


## Page 47

Dueling DQN Architecture 
The loss function remains the same  
47 


## Page 48

The Identifiability Problem 
Naïve approach: 𝑄𝑠, 𝑎= 𝑉𝑠+ 𝐴𝑠, 𝑎 
This decomposition is not unique. For any constant c: 
  
𝑉′ 𝑠= 𝑉𝑠+ 𝑐, 𝐴′ 𝑠, 𝑎= 𝐴𝑠, 𝑎−𝑐 
  
give the same Q 
The network can change the values between streams without 
changing Q(s, a) 
Training becomes unstable because the roles of V and A drift 
48 


## Page 49

Normalized Advantage 
The dueling architecture normalizes the advantages: 
  
𝑄𝑠, 𝑎= 𝑉𝑠+
𝐴𝑠, 𝑎−1
𝒜 𝐴𝑠, 𝑎′
𝑎′
 
      where |A| is the number of actions 
The advantages in each state average to 0 
V(s) captures the expected Q-value across actions 
A(s, a) represents the deviations around this average 
Subtracting the same mean from all A(s, a) preserves their ordering 
argmax over Q(s, a) matches argmax over raw advantages 
49 


## Page 50

Example: Corridor with Redundant Actions 
Consider a corridor state where many actions behave like no-ops 
Only a few actions meaningfully move the agent forward along the 
corridor 
A single-stream Q-network outputs one separate Q(s, a) value for 
each action 
It must learn low values for many useless actions individually from 
data 
Learning slows dramatically as the number of near-redundant 
actions in the state increases 
50 


## Page 51

State With Redundant Actions 
Now consider a single state s with actions a1, …, a5 
Each action from s gives reward r = 1 and then transitions to a 
terminal state 
The optimal values are identical: Q*(s, ai) = 1  ∀i 
A single-stream network has separate outputs Q(s, a1), …, Q(s, a5) for 
this state 
Each update only changes the Q(s, ai) for the selected action; the 
others may remain poorly trained for a long time 
 
51 


## Page 52

Accelerating Learning 
A dueling network produces one value V(s) and advantages  
A(s, a1), …, A(s, a5) 
It combines them as: 𝑄𝑠, 𝑎=  𝑉𝑠+  𝐴𝑠, 𝑎−
1
5  𝐴𝑠, 𝑎𝑖 
Every sample from state s directly updates the shared value V(s), 
regardless of the chosen action 
The network quickly sets V(s) ≈ 1 and keeps advantages  
A(s, ai) ≈ 0 
All Q(s, ai) ≈ 1, even for actions rarely sampled, which accelerates 
learning with redundant actions 
 
52 


## Page 53

Dueling DQN Benefits 
The dueling network separately estimates state value and action 
advantages in each state. This separation reduces sensitivity to small, 
noisy differences in estimated returns between actions 
It speeds up learning in environments that have many similar or 
redundant actions 
It improves state evaluation because the network can share 
experience from all actions in a state 
In tasks with few, always-critical actions, the dueling architecture 
provides only modest gains 
53 


## Page 54

Results 
In a synthetic corridor task with 5, 10 or 20 actions, dueling networks 
learned faster as action redundancy increased 
On 57 Atari games it improved performance in most cases, especially 
in games with many actions or few critical ones 
The dueling architecture included Double DQN training 
54 


## Page 55

Results: Human-Normalized Scores  
Game 
Double DQN 
Dueling DQN  
Atlantis 
576.1% 
2285.3% 
Krull 
592.3% 
923.1% 
Road Runner 
563.2% 
887.4% 
Star Gunner 
620.5% 
924.0% 
55 


## Page 56

Example: Atlantis 
56 


## Page 57

Deep Q-Networks 
1. Standard DQN 
2. Double DQN 
3. Dueling DQN 
4. Rainbow DQN 
57 


## Page 58

Rainbow DQN 
Combines 6, previously independent, improvements: 
Double Q-learning: overestimation 
Prioritized experience replay: important transitions 
Dueling networks: efficiency and generalization 
Multi-step learning: n-step return 
Distributional Q-learning: entire return distribution 
Noisy networks: stochastic NN layers 
58 


## Page 59

Prioritized Experience Replay 
Basic DQN samples transitions uniformly from the replay buffer 
Rainbow DQN assigns priorities based on TD error magnitude 
Large TD errors indicate surprising or underrepresented experiences 
Such transitions may deserve more frequent replay during training 
Non-uniform sampling biases the learning update away from the 
true behavior distribution 
Importance sampling downscales the losses of oversampled 
transitions to partially correct this bias 
 
Two variants: proportional or rank-based prioritization 
59 


## Page 60

Proportional Prioritization 
We define the transition priority pi using its TD error 𝛿𝑖: 𝑝𝑖= 𝛿𝑖 
Absolute value keeps priorities non-negative for all transitions 
If 𝛿𝑖= 0, that transition is never sampled 
A small є > 0 is added to avoid zero priority: 𝑝𝑖= 𝛿𝑖+ 𝜖 
Priorities are converted to probabilities: 𝑃(𝑖) =
𝑝𝑖
 𝑝𝑘
𝑘
 
We can also control the strength of prioritization: 𝑃(𝑖) =
𝑝𝑖
𝛼
 𝑝𝑘
𝛼
𝑘
 
α = 1: we strongly prioritize large pi 
α = 0: uniform random sampling 
60 


## Page 61

Rank-based Prioritization 
Rank-based prioritization defines priority from the rank of each 
transition in the replay buffer 
The replay buffer orders transitions from high TD error to low;  
Ranki is the position of transition i 
The priority of transition i is: 𝑝𝑖=
1
𝑅𝑎𝑛𝑘𝑖 
The probabilities P(i) are computed in the same way from pi 
 
61 


## Page 62

Correcting the Sampling Bias 
Proportional and rank-based prioritization over-sample transitions 
with high TD errors 
Learning then focuses on a small subset of high-error transitions, 
which increases the risk of overfitting 
Importance weights wi are used to downweight frequently sampled, 
high-priority transitions 
 
 
 
where N is the replay buffer size, P(i) is the sampling probability, and 
β gradually increases from 0.4 to 1 
 
62 


## Page 63

Multi-Step Learning 

Rainbow DQN incorporates multi-step returns into Q-learning updates for 
faster value propagation 

Sparse-reward tasks benefit because reward information travels back more 
quickly 

n-step target for starting time t: 
𝐺𝑡:𝑡+𝑛=  𝛾𝑘𝑅𝑡+𝑘+1
𝑛−1
𝑘=0
+ 𝛾𝑛max
𝑎′ 𝑄𝑆𝑡+𝑛, 𝑎′  

Replay entries correspond to length-n trajectories 
𝑆𝑡, 𝐴𝑡, 𝑅𝑡+1, … , 𝑅𝑡+𝑛, 𝑆𝑡+𝑛, not only single steps 

For each start t, we form the transition (𝑆𝑡, 𝐴𝑡, 𝐺𝑡:𝑡+𝑛, 𝑆𝑡+𝑛)  

Usually, a small n (e.g., 3) is used to preserve off-policy Q-learning while 
improving updates under delayed or sparse rewards 
 
63 


## Page 64

Distributional RL 
Classical Q-learning learns the expected return: 
  
𝑄𝑠, 𝑎= 𝔼𝐺𝑡
𝑠𝑡= 𝑠, 𝑎𝑡= 𝑎 
Different actions can have the same Q-value but have very different 
return variability 
One action can yield a steady medium reward; another can alternate 
between very high and very low rewards 
Distributional RL models the random return Z(s, a) for each  
state-action pair 
The expected value remains: 
  
𝑄𝑠, 𝑎= 𝔼𝑍𝑠, 𝑎 
  
but the learning signal comes from the full distribution 
64 


## Page 65

C51: Fixed-Atom Representation 
C51 represents each return distribution using 51 fixed atoms 
𝑧1, … , 𝑧51  
The atoms lie evenly between the bounds 𝑣min and 𝑣max or returns; 
they provide the same support for all (s, a) 
The network outputs 51 probabilities for each (s, a) 
These probabilities are updated after each transition 
65 


## Page 66

Example: Same Mean, Different 
Distributions 
Consider 2 actions in one state with one-step returns 
Action A: reward –10 with probability 0.5, +10 with probability 0.5 
Action B: reward always 0 
Both actions have the mean return equal to 0 
The standard DQN uses only expectations and cannot distinguish 
these 2 actions 
C51 learns different distributions  
Assume the atoms are {–10, 0, 10} 
Then: pA = [0.5, 0, 0.5], pB = [0, 1, 0] 
 
66 


## Page 67

Noisy Networks 
Standard DQN commonly uses є-greedy exploration for action 
selection 
A single global є ignores state-dependent uncertainty and exploration 
needs 
Preset decay schedules reduce exploration even when important 
regions remain poorly explored 
Noisy Networks add randomness to network parameters, and the 
stochastic policies become state-dependent 
67 


## Page 68

Noisy Linear Layers 
A standard linear layer computes 𝑦 =  𝑊𝑥 +  𝑏 
Noisy Networks replace this with a parameterized noisy 
transformation: 
  
𝑦= 𝑊+ 𝜎𝑊⊙𝜖𝑊𝑥+ 𝑏+ 𝜎𝑏⊙𝜖𝑏 

𝑊, 𝑏 are the usual weights and biases, but 𝜎𝑊, 𝜎𝑏 control noise 
scale or magnitude. 𝜖𝑊, 𝜖𝑏 are random variables 
Learning adjusts these scales σ, so the agent decides itself where 
strong or weak randomness is useful 
68 


## Page 69

Adaptive Exploration 
Rainbow DQN replaces standard final linear layers in the value 
network with noisy layers 
External є-greedy exploration is removed or reduced to a small 
constant if desired 
In familiar states, learning drives σ towards 0; the policy becomes 
effectively deterministic 
In uncertain or high-impact regions, σ can remain large to keep 
exploration active 
Exploration adapts automatically, guided by the same loss that trains 
the value distribution 
69 


## Page 70

Example: Two-Armed Bandit 
Let’s consider a two-armed bandit with actions 1 and 2, each with an 
unknown reward 
A small neural network takes a constant input and produces 2 
outputs through a noisy final layer 
Early in training, the learned scales σ are large, so sampled Q-values 
differ between forward passes 
Some samples rank action 1 higher, others rank action 2 higher, so 
both arms are explored 
As one action’s expected value becomes clearly better, gradients push 
its corresponding σ toward smaller values 
70 


## Page 71

Example: Corridor Junction 
The environment is a corridor with a junction: the left branch gives a 
small immediate reward, the right a delayed (distant) large reward 
є-greedy exploration often breaks long right-branch runs because 
random action flips interrupt the trajectory 
A sampled noisy-parameter set tends to stay roughly fixed over many 
steps, sometimes for an entire episode 
One such sample can consistently prefer right-branch actions, 
allowing the agent to reach the distant large reward 
The observed large reward then updates both the base weights and 
the noise scales to make similar coherent trajectories more likely in 
the future 
71 


## Page 72

Results: Human-Normalized Scores  
Game 
Dueling DQN 
Rainbow DQN 
Alien 
67% 
134% 
H.E.R.O. 
68% 
184% 
Ice Hockey 
79% 
102% 
Yars’ Revenge 
45% 
193% 
Montezuma’s Revenge 
0% 
8% 
72 


## Page 73

Example: Montezuma’s Revenge 
73 


## Page 74

Atari Games after DQN 
OpenAI using PPO (2018): 1570% of average human performance for 
Montezuma’s Revenge  
Uber AI Go-Explore (2019): 920% for Montezuma’s Revenge without 
domain knowledge, 14 000% with domain knowledge 
DeepMind Agent57 (2020): First agent better than humans on all 57 
Atari games; 200% for Montezuma’s Revenge 


## Page 75

Conclusions 
Standard DQN demonstrates end-to-end control from pixels but 
suffers from overestimation and unstable, sample-inefficient learning 
Double DQN reduces value overestimation, stabilizes training, and 
usually achieves better performance than standard DQN 
Dueling DQN separates state value and action advantage, which 
improves generalization and learning speed in states with redundant 
actions 
Rainbow DQN combines six extensions into one architecture, and 
delivers the strongest, most robust Atari results among DQN variants 
75 


## Page 76

Main References 
Ravichandiran, S. (2020). Deep Reinforcement Learning with 
Python. 2nd edition. Packt Publishing, Birmingham, UK 
Hessel, M., et al. (2018). Rainbow: Combining Improvements in 
Deep Reinforcement Learning. Proceedings of AAAI-18. AAAI 
Press, New Orleans, LA, USA. https://arxiv.org/pdf/1710.02298 
 
 
76 
