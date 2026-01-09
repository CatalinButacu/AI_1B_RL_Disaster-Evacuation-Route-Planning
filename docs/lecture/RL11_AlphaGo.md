# RL11_AlphaGo

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL11_AlphaGo.pdf

**Pages:** 48

---


## Page 1

Reinforcement Learning 
11. The AlphaGo Family 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

The AlphaGo Family 
1. AlphaGo 
2. AlphaGo Zero 
3. AlphaZero 
4. MuZero 
2 


## Page 3

The AlphaGo Family 
1. AlphaGo 
2. AlphaGo Zero 
3. AlphaZero 
4. MuZero 
3 


## Page 4

Go 

19×19 board with 361 intersections 

Size of state space: 10170 

Early and middle game states often offer 100–200 legal moves, far above chess 

Games typically last several hundred moves 

Reward is almost entirely delayed: a single terminal signal, +1 win or –1 loss 
Atari 
4 


## Page 5

Why Go Is Difficult to Learn 
Random or near-random games teach almost nothing about good 
local tactics or global strategy 
Credit assignment horizons are very long (hundreds of moves); 
crucial early exchanges can determine results far later 
TD learning can in principle backpropagate value, yet noisy 
intermediate decisions introduce massive variance 
The value function should unify local capture patterns with global 
territory, influence, and balance; small local changes can flip results 
5 


## Page 6

Monte Carlo Tree Search 
The difficulty of Go motivates explicit lookahead: players imagine 
concrete futures instead of relying on value approximation 
Monte Carlo Tree Search (MCTS) concentrates simulations on 
promising or uncertain actions, guided by statistics from previous 
rollouts 
Early Go engines paired MCTS with hand-crafted pattern rules, 
heuristic rollouts, and static territory or influence evaluations 
These systems reached strong amateur level but missed subtle long-
term sacrifices and global trade-offs 
6 


## Page 7

AlphaGo: Overall Architecture 
AlphaGo combines deep policy networks, a value network, and MCTS 
It uses Go-specific input features and learned representations to 
guide search efficiently 
It separates what to play (policy) from who is winning (value) 
It relies on both human expert data and self-play  
It is a solution between early hand-crafted Go programs and later, 
more general successors 
7 


## Page 8

Policy and Value Networks 
8 


## Page 9

Input Features 
The Go board is seen as a 19×19 grid with multiple feature planes 
The feature planes include black stones, white stones, liberties, 
captures, move history, etc. 
For each position on the game board there are 48 binary or integer 
features, for example: 

Whether the position is occupied by AlphaGo, by the opponent, or is empty 

The number of adjacent empty positions 

The number of the opponent’s stones that would be captured by placing a stone 
on that position 

How many moves ago a stone was placed there, etc. 
Stacked planes form an image-like tensor input to convolutional 
layers 
9 


## Page 10

Training 
10 


## Page 11

1. Supervised Policy Network 
The first component is a supervised policy network pσ ​(a ∣ s) trained 
to imitate expert human moves 
Input: encoded Go position s 
Output: probabilities over legal moves a 
Trained on professional game records using cross-entropy loss (with 
a one-hot expert target): 
 
  
Learns a strong human-like prior over plausible moves in many 
positions 
Can play at strong amateur level without additional search 
11 


## Page 12

2. RL Policy Network 
The second component is a reinforcement learning policy network 
pρ ​(a ∣ s) initialized with the supervised policy parameters pσ 
The 𝑝𝜌 is improved through self-play: two copies of the current 
policy network play full games against each other (no MCTS) 
Each game receives a return z = +1 for win and z = –1 for loss 
The policy is updated with REINFORCE-style loss: 
 
 
The moves from winning games become more probable, and vice 
versa. The shared weights are updated (no separate teacher-student) 
This helps the network to surpass human performance 
12 


## Page 13

3. Value Network 
The third component is a value network that predicts a winning 
probability (for the game) outcome from a position s: 𝑣𝜃𝑠∈−1,1  
It is trained on self-play positions with the known final result  
𝑧∈{−1, +1}  
It uses MSE regression loss: 
  
𝐿𝑣𝑎𝑙𝑢𝑒= 𝑣𝜃𝑠−𝑧2 
13 


## Page 14

4. Rollout Policy Network 
The fourth component is the rollout policy network: a small, shallow 
convolutional network trained from expert games, used only inside 
MCTS rollouts during search 
It generates fast, plausible (but not necessarily smart) moves to 
replace random play; this reduces rollout variance and improves value 
estimates 
 
The final leaf evaluation uses both rollout and value: 
  
𝑉𝑙𝑒𝑎𝑓= 1 −𝜆 𝑧𝑟𝑜𝑙𝑙𝑜𝑢𝑡+ 𝜆 𝑣𝜃𝑠 
14 


## Page 15

5. MCTS 
During gameplay, AlphaGo uses MCTS augmented with policy and 
value networks 
At each node, the policy network initializes the priors 𝑃𝑎𝑠) over 
child moves 
The leaf nodes are evaluated using the blended estimate Vleaf 
Action selection uses PUCT (Policy-UCB applied to Trees): 
15 


## Page 16

Alpha Go 

The supervised policy network 𝑝𝜎 is trained on expert games 

The RL policy 𝑝𝜌 is initialized from 𝑝𝜎 and improves via self-play 

Self-play games are generated using 𝑝𝜌 and the value network 𝑣𝜃 is 
trained on the game results 

A fast rollout policy is trained from human games for efficient playouts 

The networks and MCTS are combined into the final AlphaGo system 
 

Alpha Go demonstrates that deep neural networks can provide strong 
priors and evaluations for search in complex games 

It still depends on human data and Go-specific features 
 

Google DeepMind: AlphaGo - The Movie 
https://www.youtube.com/watch?v=WXuK6gekU1Y 
16 


## Page 17

The AlphaGo Family 
1. AlphaGo 
2. AlphaGo Zero 
3. AlphaZero 
4. MuZero 
17 


## Page 18

AlphaGo Zero 
It learns Go from scratch using only the game rules, without human 
games or features 
It maintains the AlphaGo skeleton: deep networks + MCTS 
It uses a single network for both move selection and position 
evaluation 
Self-play games provide all training data 
It can be viewed as approximate policy iteration:  
network → MCTS → self-play data → network update → repeat 
18 


## Page 19

Unified Policy-Value Network 
The network maps a Go state s to both policy and value: 
   
𝑓𝜃𝑠= 𝑝, 𝑣 
p: the probability distribution over all moves, including pass 

𝑣∈−1,1 : the predicted win probability for the current player 
AlphaGo Zero uses a residual network (ResNet) instead of a simple 
convolutional network 
 
 
 
 
 
The network has a shared part and two heads (for p and v) 
 
ResNet illustration, not AlphaGo Zero 
19 


## Page 20

Input Encoding 
The board is represented as a 19×19×17 tensor of binary feature 
planes 
8 planes for current player’s stones over the last 8 time steps 
8 planes for opponent’s stones over the last 8 time steps 
1 plane indicating which color is to move 
No explicit domain-specific tactical features 
20 


## Page 21

Architecture 


## Page 22

MCTS Steps 
At a node, AlphaGo Zero calls the network 𝑓𝜃 for p and v 
There is no rollout policy; the value head v replaces the random 
playouts entirely 
The priors p initialize new child edges in the tree 
The value v backs up through the path to update the Q estimates 
The PUCT action selection rule is used 
The root policy is updated from the visit counts: 
  
𝜋𝑎
𝑠0
∝𝑁𝑠0, 𝑎1/𝜏 
τ is the exploration temperature 
τ is higher early in training to encourage exploration, and near 0 later so the 
agent picks the most visited move 
22 


## Page 23

Self-Play Data Generation 
AlphaGo Zero plays games against itself using MCTS and the current 
network 
At each move, it samples the action from root policy 𝜋⋅𝑠𝑡 
The game ends with an outcome 𝑧∈{−1, +1} 
For each timestep t, training tuple 𝑠𝑡, 𝜋𝑡, 𝑧𝑡 are stored 
The data encode both search behavior and true long-term outcomes 
A replay buffer accumulates many such triplets from many games 
23 


## Page 24

Training Targets 
st is the encoded position seen by the network during play 
πt is the improved policy from MCTS visit counts at state st  
zt is the final game result from the perspective of player to move at t 
πt acts as a stronger teacher than raw network policy 
zt provides a ground truth signal for long-horizon value prediction 
24 


## Page 25

Loss Function 
For one position, the network outputs 𝑝𝑡, 𝑣𝑡= 𝑓𝜃𝑠𝑡 
AlphaGo Zero uses a combined loss: 
  
𝐿𝜃= 𝑧−𝑣2 − 𝜋(𝑎|𝑠)
𝑎
⋅log 𝑝(𝑎|𝑠) + 𝑐𝜃2 
The first term: squared error that drives the value vt toward the 
game outcome zt 
The second term: cross-entropy that pushes policy pt toward the 
search policy πt 
The regularization term 𝑐𝜃2 stabilizes training and discourages 
overfitting 
25 


## Page 26

Outer Training Loop 
Step 1: Generate self-play games using the current best network θbest 
and MCTS 
Step 2: Collect 𝑠𝑡, 𝜋𝑡, 𝑧𝑡 into the replay buffer 
Step 3: Train the network with gradient descent on batches from the 
buffer and obtain an updated network θnew​ 
Step 4: Evaluate θnew  versus the current best θbest in direct matches 
If the new network wins more than some threshold (e.g., 55%), promote it 
to be the new best 
Otherwise keep the old best network and continue training 
26 


## Page 27

AlphaGo Zero vs. AlphaGo 
AlphaGo Zero removes the need for human game data; learning 
relies entirely on self-play 
It discards handcrafted Go features and uses only stone histories and 
side-to-move 
It eliminates the rollout policy; the value head provides all leaf 
evaluations 
It uses a single network for both policy and value, which simplifies 
the architecture 
It represents a cleaner, more general template for search-guided 
reinforcement learning 
27 


## Page 28

Performance 
28 


## Page 29

The AlphaGo Family 
1. AlphaGo 
2. AlphaGo Zero 
3. AlphaZero 
4. MuZero 
29 


## Page 30

AlphaZero 
AlphaZero is based on the AlphaGo Zero architecture 
It mastered Go, chess, and shogi (Japanese chess) from scratch using 
only self-play and the game rules, with no human examples 
AlphaZero uses the same architecture and learning algorithm for all 
three games:  
A deep residual neural network with 19 blocks  
Shared weights for both policy and value outputs  
Monte Carlo Tree Search 
The input encoding depends on the game 
For Go, AlphaZero uses the same 17-plane encoding as AlphaGo Zero 
30 


## Page 31

Input Encoding: Chess 

119 binary planes: 

Planes for each piece type and color over a short history, e.g., where white pawns 
were for several recent moves, where the black queen was, etc. 

Extra planes for castling rights, side to move, move counters (like the fifty-move 
rule), and similar rule-related information 
31 
8×8 board 


## Page 32

Input Encoding: Shogi 

Different rules: captured pieces can be reinserted into the game (by the 
captor), promotions differ from chess, piece moves are also slightly different 

The encoding expands to 362 planes, including piece-in-hand information 
and promotion-related flags 
32 
9×9 board 


## Page 33

Handling Draws 
Go training in AlphaGo Zero considers only win or loss outcomes 
Chess and, to a lesser extent, shogi require incorporating draws and 
repetition-related termination rules 
The value head must represent {–1, 0, +1} as outcomes 
Self-play can naturally include draws; no special-case heuristic is 
needed 
33 


## Page 34

Performance 
AlphaZero defeated Stockfish, the strongest traditional chess engine, 
after 4 hours of self-play training 
It beat Elmo, a top shogi engine, after 2 hours of training  
AlphaZero searched far fewer nodes than Stockfish or Elmo in chess 
and shogi, because it can focus search on promising paths 
 
Its playstyle is often aggressive and unconventional compared to 
traditional engines 
34 


## Page 35

AlphaZero vs. AlphaGo Zero 

In Go, AlphaZero surpassed AlphaGo Zero after 24 hours of training, 
despite reusing the same architecture 

AlphaZero no longer used the evaluation phase (θnew  vs. θbest); it 
continuously updated the same network with the latest self-play data 

It used a batch size of 4096, double the 2048 used by AlphaGo Zero 

DeepMind used 5000 first-generation TPUs for generating self-play 
games and 64 second-generation TPUs for training 

In its training run, AlphaZero processed ~21 million games, compared to 
the ~5 million games processed by AlphaGo Zero 
35 


## Page 36

Significance  
AlphaZero demonstrates that the architecture is not tied to Go 
It establishes a reusable template for deterministic, perfect-
information board games 
Only the encodings and rules simulators are hand-designed 
Everything else (strategies, evaluations, style of play) emerges from 
self-play with a single algorithm 
36 


## Page 37

The AlphaGo Family 
1. AlphaGo 
2. AlphaGo Zero 
3. AlphaZero 
4. MuZero 
37 


## Page 38

MuZero: Unknown Transition Model 
Alpha* models assume that a perfect simulator exists for every move 
Many real domains lack explicit, hand-coded transition functions 
MuZero keeps the Alpha* structure: policy-value network, MCTS, 
and self-play, but no longer needs a known transition model 
The search runs inside a learned model instead of a hand-written 
engine 
38 


## Page 39

Predicting only the Essentials for Decisions 
Classic world models often reconstruct next full observations, e.g., 
images 
This reconstruction forces care about many details irrelevant to 
decisions 
MuZero focuses only on predicting rewards, values, and good policies 
The model may be wrong about irrelevant future aspects without 
penalty 
The main criterion is to preserve decision quality, not observation 
fidelity 
39 


## Page 40

Three-Component Model 
The representation network ℎ𝜃: 𝑂∗→𝑆 maps an observation history 
to the initial latent state s0 
The dynamics network 𝑔𝜃: 𝑆× 𝐴→𝑆× ℝ maps (sk , ak) to the next 
latent state and reward 
The prediction network 𝑓𝜃: 𝑆→Δ(𝐴) × ℝ maps a latent state sk to 
policy and value 
All three networks share parameters θ and train jointly 
Together they form an internal “environment” where planning 
occurs 
 
40 


## Page 41

Representation Network 
The representation network is responsible for the construction of 
hidden states 
The input is the recent observation history 𝑜1:𝑡 from the real 
environment 
The representation network computes the root latent state: 
𝑠0 = ℎ𝜃𝑜1:𝑡 
s0 does not need to resemble images or boards directly 
MuZero only needs information sufficient for predicting rewards, 
values, and policies 
The model learns its own abstractions, like threats or configurations, 
inside this latent space 
41 


## Page 42

Dynamics Network 
The dynamics network is responsible for the construction of latent 
transitions and rewards 
It takes a latent state and action 𝑠𝑘, 𝑎𝑘 and outputs the next latent 
state and the predicted immediate reward: 
  
𝑠𝑘+1, 𝑟𝑘= 𝑔𝜃𝑠𝑘, 𝑎𝑘 
Repeated applications unroll imagined trajectories entirely in the 
latent space 
There are no calls to the real environment during the search; only 
𝑔𝜃 generates the “futures” 
Reward prediction trains the model to represent decision-relevant 
consequences of actions 
42 


## Page 43

Prediction Network 
The prediction network assesses the policies and values 
For any latent state sk , the prediction network outputs 
  
𝑝𝑘, 𝑣𝑘= 𝑓𝜃𝑠𝑘 

pk : policy distribution over actions for that imagined state 

vk : value estimate of long-term return from that state 
It provides priors and leaf evaluations for tree search inside the 
model 
43 


## Page 44

Collecting Training Data from Real Episodes 
MuZero interacts with the real environment to generate trajectories 
Each episode yields observations ot , actions at , and actual rewards 
from the real environment ut 
Self-play for games; standard RL interaction for Atari-style tasks 
It stores full sequences 𝑜1, 𝑎1, 𝑢1, 𝑜2, … , 𝑜𝑇 in a replay buffer 
Later it picks random time indices t as roots for training unrolls 
44 


## Page 45

Learning by Unrolling the Model 
For a chosen root time t, compute 𝑠0 = ℎ𝜃𝑜1:𝑡 
Unroll model K steps using recorded actions: 𝑠𝑘+1, 𝑟𝑘= 𝑔𝜃𝑠𝑘, 𝑎𝑡+𝑘 
At each step k, apply 𝑓𝜃𝑠𝑘 to obtain 𝑝𝑘, 𝑣𝑘 
Reward rk  target: actual reward 𝑢𝑡+𝑘 from the environment 
Value vk target: the truncated return 
  
𝑧𝑡+𝑘≈𝑢𝑡+𝑘+ 𝛾∙𝑢𝑡+𝑘+1 + ⋯+ 𝛾𝑛∙𝑢𝑡+𝑘+𝑛+ 𝛾𝑛+1𝑣  
45 


## Page 46

MuZero Operations 
46 


## Page 47

Conclusions 
The AlphaGo family moved RL from toy worlds and classic Atari 
games into world-class decision making 
AlphaGo succeeded in beating a human world champion in Go using 
human data, deep networks, and MCTS 
AlphaGo Zero learned Go from scratch, without human games 
AlphaZero generalized for different games 
MuZero generalized further for unknown environment dynamics 
47 


## Page 48

Main References 
Silver, D., et al. (2016). Mastering the Game of Go with Deep Neural 
Networks and Tree Search. Nature 
Silver, D., et al. (2017). Mastering the Game of Go without Human 
Knowledge. Nature 
Silver, D., et al. (2018). A General Reinforcement Learning Algorithm 
that Masters Chess, Shogi, and Go through Self-Play. Science 
Schrittwieser, J., et al. (2020). Mastering Atari, Go, Chess and Shogi 
by Planning with a Learned Model. Nature 
48 
