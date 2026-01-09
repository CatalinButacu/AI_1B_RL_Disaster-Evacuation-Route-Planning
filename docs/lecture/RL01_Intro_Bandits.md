# RL01_Intro_Bandits

**Source:** c:\Users\catalin.butacu\Downloads\RL\resourses\lecture\RL01_Intro_Bandits.pdf

**Pages:** 89

---


## Page 1

Reinforcement Learning 
1. Introduction. Multi-Armed Bandits 
 
Florin Leon 
 
“Gheorghe Asachi” Technical University of Iași, Romania 
Faculty of Automatic Control and Computer Engineering 
 
https://florinleon.byethost24.com/lect_rl.html 
 
2025 
 


## Page 2

Introduction. Multi-Armed Bandits 
1. Introduction to Reinforcement Learning 
 
1.1. RL Among Machine Learning Paradigms 
 
1.2. Key Elements of RL 
 
1.3. Related Fields 
2. Multi-Armed Bandit Problems 
 
2.1. Sample-Average Estimation of Action Values 
 
2.2. Exponential Recency-Weighted Average 
 
2.3. Optimistic Initial Estimates 
 
2.4. Upper-Confidence-Bound Action Selection 
 
 
2 


## Page 3

Introduction. Multi-Armed Bandits 
1. Introduction to Reinforcement Learning 
 
1.1. RL Among Machine Learning Paradigms 
 
1.2. Key Elements of RL 
 
1.3. Related Fields 
2. Multi-Armed Bandit Problems 
 
2.1. Sample-Average Estimation of Action Values 
 
2.2. Exponential Recency-Weighted Average 
 
2.3. Optimistic Initial Estimates 
 
2.4. Upper-Confidence-Bound Action Selection 
 
 
3 


## Page 4

Introduction 
Reinforcement learning (RL) focuses on learning through interaction 
with the environment, with no explicit teacher 
The agent learns by trial and error 
Natural learning examples: riding a bicycle, language acquisition, 
driving. Here, there is some kind of supervision, but one cannot 
succeed without direct practice 
4 


## Page 5

Machine Learning Paradigms 
Machine learning (ML) has three main paradigms:  
Supervised learning 
Unsupervised learning 
Reinforcement learning 
Each paradigm has different learning methods, based on 
available information 
RL is a sequential decision-making approach 
5 


## Page 6

Machine Learning Paradigms 
Supervised learning: learn a function from labeled data 
Unsupervised learning: find patterns without labels 
Reinforcement learning: learn by interacting with the 
environment to maximize rewards 
6 


## Page 7

Learned Functions 
Some functions are known exactly, e.g., Newton’s second law of 
motion: F = m a 
But they were previously induced and confirmed experimentally 
Many functions need to be learned from data, e.g., predicting 
shampoo purchases from age 
Given age, predict 0 or 1 
They may have no known analytical expressions 
E.g., complex polymerization reactions 
In such cases, ML methods are used to approximate the functions 
7 


## Page 8

Supervised Learning 
Classification (discrete output) 
 
 
 
 
Regression (numerical output) 
 
 
 
 
8 


## Page 9

Supervised Learning 
In supervised learning, the algorithm learns from individual 
examples given as labeled data: pairs of inputs x and outputs  
y = f(x) 
The algorithm learns an approximation 𝑓  of the (possibly very 
complex) function  
There are many types of algorithms for supervised learning, e.g., 
decision trees, probabilistic, instance-based, neural networks (NNs) 
When using NNs, common loss functions include cross-entropy (for 
classification) and mean squared error (for regression) 
9 


## Page 10

Unsupervised Learning 
Unsupervised learning works with data that has no labels 
The goal is to discover patterns in the data, such as clusters or 
subgroups 
Techniques include k-means, Expectation-Maximization (EM), DBSCAN 
10 


## Page 11

Unsupervised Learning Directions 
Clustering 
Grouping similar data points, e.g., customer segmentation 
Dimensionality reduction 
Reducing feature space for easier analysis, e.g., PCA or t-SNE for 
visualization 
Representation learning 
Learning efficient, lower-dimensional representations of data that 
retain essential information., e.g., autoencoders 
11 


## Page 12

Clustering Applications 

Customer segmentation 

Used in marketing and sales to group customers based on purchasing behavior, 
demographics, or engagement 

Helps with targeted advertising, product recommendations, and personalized 
experiences 

Document clustering 

Organizes large volumes of unstructured text, e.g., news articles, research papers, legal 
documents 

Used in search engines, digital libraries, and topic discovery 

Social network analysis 

Clusters people or nodes based on connection patterns, e.g., communities, influence 
groups 

Applied in marketing, misinformation detection, and online behavior analysis 

Urban planning and geospatial analysis 

Clusters locations based on factors like traffic patterns, population density, or land use  

Used for zoning, resource allocation, and emergency planning 
 
12 


## Page 13

Reinforcement Learning 
RL agents learn by interacting with the environment, rather than 
from a static dataset 
The goal is to find an optimal policy that maps states to actions to 
maximize long-term cumulative rewards 
In supervised and unsupervised learning, the full dataset is usually 
given 
But continual, online learning methods, exist as well 
RL learns step by step, as the agent receives feedback (rewards) from 
the environment 
13 


## Page 14

Reinforcement Learning 
The agent adjusts its behavior (actions) to maximize total rewards 
over time 
Actions affect both immediate and future rewards 
Delayed consequences play an essential role in determining optimal 
actions 
Environments are often complex, non-deterministic, and dynamic 
14 


## Page 15

RL Applications 
Operating adaptive controllers in refineries 
Optimizing energy use in power grids 
Autonomous driving 
Suggesting a medical treatment plan 
Making hold-buy-sell decisions in trading 
15 


## Page 16

Games 

Games have long been used to study 
intelligent decision making in simple, 
controlled environments 

Board games: Backgammon, Go 

Video games: Atari suite, Pac-Man, 
StarCraft (multi-agent) 

RL strategies improve through repeated 
gameplay 


## Page 17

Robotics 
Robots can be pre-programmed for specific tasks, but are 
limited in adaptability 
Developers may struggle to describe operational knowledge, 
e.g., how “muscles” move when picking up an object 
RL allows robots to learn from experience and adapt to new or 
changing environments 
Tasks like navigation, manipulation, and locomotion can be 
learned through RL, not pre-programmed 
17 


## Page 18

Examples 
Robot flipping pancakes 
Autonomous model helicopter  
 
18 


## Page 19

Reinforcement Learning vs.  
Supervised/Unsupervised Learning 
In supervised learning, the correct output value is provided 
In unsupervised learning, there is no given value 
In RL, the agent must explore to discover optimal actions based 
on rewards (~“good” or “bad”) 
An intermediate case in terms of available information 
RL requires making decisions without knowing the exact 
outcomes, whereas supervised learning assumes fixed data 
Rewards may be sparse, e.g., a game was won or lost 
19 


## Page 20

Exploration and Exploitation 
In RL, an agent must decide between exploring new actions and 
exploiting known actions with high returns (total rewards) 
Exploration involves trying new actions to gather more information 
about the environment 
Exploitation focuses on taking actions that have previously resulted 
in high returns 
The challenge is to balance these two strategies to avoid either over-
exploring or sticking too much to known actions 
Too much exploration: low returns  
Too much exploitation: suboptimal results 
20 


## Page 21

The Exploration-Exploitation Dilemma: 
Examples 
Do you go to the restaurant you’ve known and liked for a long time, 
or try the newly opened one?  
Does it make a difference if you’re in your hometown or in another city 
where you’re staying for only two nights? 
Do you go with your best friend or with someone you’d like to get to 
know better? 
Do you order a familiar dish or try something new? 
 
Companies invest money in research and development to invent new 
products (e.g., medications), but they also want to profit from 
existing production lines that are already successful. How much 
money should they invest in research? 
21 


## Page 22

Introduction. Multi-Armed Bandits 
1. Introduction to Reinforcement Learning 
 
1.1. RL Among Machine Learning Paradigms 
 
1.2. Key Elements of RL 
 
1.3. Related Fields 
2. Multi-Armed Bandit Problems 
 
2.1. Sample-Average Estimation of Action Values 
 
2.2. Exponential Recency-Weighted Average 
 
2.3. Optimistic Initial Estimates 
 
2.4. Upper-Confidence-Bound Action Selection 
 
 
22 


## Page 23

Rewards 
A reward is the immediate feedback that defines what is 
desirable or undesirable in the environment 
Rewards are the indicators of success or failure in achieving the 
goal 
The agent’s objective is to maximize the total reward over time, 
not just immediate gains 
Rewards can be stochastic, influenced by both the agent’s 
actions and the environment’s state 
The sum of rewards over time is called the return 
23 


## Page 24

Value Function 
The value function estimates the expected return that can be 
obtained starting from a given state 
It helps agents evaluate which states are worth pursuing based 
on their future rewards 
Agents make decisions based on long-term value rather than 
immediate reward 
Estimating values is difficult because future states and rewards 
are uncertain 
24 


## Page 25

Policy 
The policy determines how the agent behaves by mapping 
states to actions 
A policy can be deterministic or stochastic, where action 
probabilities are assigned to states 
Deterministic: 𝜋: 𝑆→𝐴, 𝜋𝑠= 𝑎 (a is the action taken in state s) 
Stochastic: 𝜋: 𝑆× 𝐴→0, 1 , 𝜋𝑎|𝑠 is the probability of taking a in s  
The agent updates the policy over time, seeking actions that 
yield the highest cumulative rewards 
 
This is the goal of an RL problem – the policy encapsulates the 
agent’s learned behavior 
 
25 


## Page 26

Environment Model 
The model represents the transition probability of the 
environment between successive states, given an action 
Model-free methods rely solely on trial and error 
Some RL systems use a model to predict the environment’s 
response to actions 
Model-based methods incorporate planning into learning by 
simulating potential future states before taking actions 
26 


## Page 27

Introduction. Multi-Armed Bandits 
1. Introduction to Reinforcement Learning 
 
1.1. RL Among Machine Learning Paradigms 
 
1.2. Key Elements of RL 
 
1.3. Related Fields 
2. Multi-Armed Bandit Problems 
 
2.1. Sample-Average Estimation of Action Values 
 
2.2. Exponential Recency-Weighted Average 
 
2.3. Optimistic Initial Estimates 
 
2.4. Upper-Confidence-Bound Action Selection 
 
 
27 


## Page 28

Psychology 
The term reinforcement learning was first used in the 
psychology literature, not in computer science 
E. Thorndike (1898) proposed the Law of Effect: actions followed by 
rewards become more likely 
B. F. Skinner (1930s–1950s) introduced operant conditioning: 
behavior changes through rewards and punishments 
C. Hull (1943) built a mathematical theory of learning: reinforcement 
is a quantifiable variable that increases the likelihood of a behavior 
in response to a stimulus 
28 


## Page 29

Conditioning: Pavlov’s Dog 

(1) a dog salivates when seeing food, (2) but initially not when hearing a bell,  
(3) when the sound rings often enough together when food is served, the dog starts 
to associate the bell with food, and (4) also salivates when only the bell rings  
29 


## Page 30

Conditioning  
30 


## Page 31

Neuroscience 
The brain learns from surprises 
Dopamine signals go up when things are better than expected, and 
down when worse, like reward prediction errors in RL 
The brain ignores cues that add no new information 
Blocking: no learning happens if one signal already predicts reward, 
like in temporal-difference learning 
The brain also learns from predictors of predictors 
Higher-order conditioning: a cue gains meaning by predicting 
another cue, like bootstrapped learning 
31 


## Page 32

Neuroscience 
The brain updates expectations gradually over time 
Dopamine changes track small differences between expected and 
actual outcomes, like temporal-difference learning adjusts 
predictions step by step in RL 
The brain separates evaluation from decision-making 
One system estimates how good things are (critic), another decides 
what to do (actor), like actor-critic methods in RL 
The brain keeps short-term memory of recent actions 
When a reward arrives, it strengthens recent brain activity, like 
eligibility traces reinforce helpful past steps in RL 
32 


## Page 33

Mathematics  
Probabilities are an integral part of the formalization of 
reinforcement learning problems 
Markov Decision Processes (MDPs) define the framework for 
modeling RL environments 
Expected values are used to evaluate and compare actions or policies 
under uncertainty 
Continuous optimization methods, such as gradient descent, 
are essential in deep reinforcement learning for training neural 
networks 
33 


## Page 34

Optimal Control 
Both RL and optimal control focus on controlling dynamical 
systems 
Key concepts, such as Bellman’s equations, were proposed in 
the context of optimal control theory within the field of 
automatic control 
RL and optimal control have different terminologies but solve 
similar sequential decision problems 
34 


## Page 35

Introduction. Multi-Armed Bandits 
1. Introduction to Reinforcement Learning 
 
1.1. RL Among Machine Learning Paradigms 
 
1.2. Key Elements of RL 
 
1.3. Related Fields 
2. Multi-Armed Bandit Problems 
 
2.1. Sample-Average Estimation of Action Values 
 
2.2. Exponential Recency-Weighted Average 
 
2.3. Optimistic Initial Estimates 
 
2.4. Upper-Confidence-Bound Action Selection 
 
 
35 


## Page 36

Introduction to Multi-armed Bandits 
Reinforcement learning uses evaluative feedback, not 
instructive feedback 
Instructive feedback (used in supervised learning) tells the agent the 
correct response for a given situation 
Evaluative feedback: evaluates actions based on their outcome, but 
does not specify the correct action 
Evaluative feedback tells the agent how good an action is, not what 
the best action is 
This leads to the need for active exploration 
The agent must try different actions and learn from the outcomes 
 
 
36 


## Page 37

Bandits – Slot Machines 

The term one-armed bandit originates 
from the design and reputation of early 
slot machines 

Early slot machines featured a single 
lever on one side that players pulled to 
start the game 

This lever looked much like a human 
arm 

The machines earned a reputation for 
keeping players’ coins, which led people 
to compare them to a bandit 

The combination of a single arm and the 
notion of stealing money resulted in the 
nickname one-armed bandit 


## Page 38

The k-Armed Bandit Problem 
The k-armed bandit problem models decision making with 
multiple options, i.e., k actions 
Actions are analogous to playing levers on a slot machine  
(a bandit) 
Each action has an expected reward 
The goal is to maximize total reward over time 
The multi-armed bandit problem is a simplified RL scenario 
that helps study exploration vs. exploitation 
38 


## Page 39

Expected Value 
The expected value of a random variable X is the average value we 
would get if we sampled X many times 
If X is a discrete random variable with probability mass function 
𝑃(𝑋 = 𝑥𝑖): 
 
 
If X is a continuous random variable with probability density 
function p(x), an integral is used, but in general, we will use the 
discrete form 
The expected value can be estimated from N samples x1, …, xN : 
 
 
This sample average approximates the true expected value as N 
becomes large 
[
]
(
)
i
i
i
X
x
P X
x




1
1
[
]
N
i
i
X
x
N



39 


## Page 40

Example: Expected Value 
In a bag with a large number of marbles, 30% are red and 70% 
are blue 
If you draw a red marble, you gain 1 point; if you draw a blue 
marble, you gain 4 points 
What is your expected gain? 
𝑥1 = 1 with probability 𝑃(𝑋 =  1) =  0.3 
𝑥2 = 4 with probability 𝑃(𝑋 =  4) =  0.7 
Expected value: 
 
Would you pay 300 points to play this game 100 times? 
 
[
]
1 0.3
4 0.7
0.3
2.8
3.1
X 




40 


## Page 41

Example: Sample Average 
Suppose we draw 10 samples: 4, 4, 1, 4, 4, 1, 4, 4, 4, 1 
Sample average: 
 
 
 
The sample-based estimate of the expected value is 3.5, which 
is close to the true value 3.1, but not exact due to the limited 
number of samples 
1
35
[
]
(4
4
1
4
4
1
4
4
4
1)
3.5
10
10
X 









41 


## Page 42

Defining the k-Armed Bandit Problem 
The agent repeatedly chooses among k different actions 
Each action provides a numerical reward, drawn from a 
stationary probability distribution 
A probability distribution that remains unchanged over time 
The goal is to maximize the total expected reward over a given 
time period (e.g., 1000 action selections) 
Real-world examples:  
A doctor selecting treatments for patients 
A company deciding between advertising campaigns 
 
42 


## Page 43

Example: Clinical Trials 

A doctor has k treatments to compare 

Each treatment i has an unknown probability of success pi​ 

T patients are enrolled sequentially 

After treating each patient, the doctor observes an outcome (success or 
failure) 

The objective is to maximize the total number of successes by sequentially 
choosing the best treatment 

At each step, the doctor chooses a treatment i 

Exploitation: he assigns patients to the treatment that appears most 
effective 

Exploration: he assigns patients to lesser-known treatments to learn more 
43 


## Page 44

Exploration vs. Exploitation 
2 actions / treatments / choices 
The (unknown) distributions, i.e., the success probabilities in 
general: p1 = 0.6, p2 = 0.8 
Trials: action (1 or 2) – outcome (1 success or 0 failure) 
1 – 0; 2 – 0; 1 – 1; 2 – 0; 1 – 1; 2 – 1 
The estimated probabilities so far: 𝑝 1 = 2/3, 𝑝 2 = 1/3 
Exploitation: keep selecting action 1 
Exploration: try action 2 
Without exploration, the optimal action (2) cannot be found 
44 


## Page 45

k-Armed Bandits: Expected Reward and 
Action Values 
Each action a (out of k possible) has an expected reward (true 
action value): 𝑞∗(𝑎) = 𝔼[𝑅𝑡∣𝐴𝑡= 𝑎] 
If 𝑞∗(𝑎) were known, the optimal strategy would be to always 
select 𝑎∗= argmax
𝑎
𝑞∗(𝑎) 
argmax returns the action a that gives the highest estimated value 𝑞∗(𝑎) 
However, 𝑞∗(𝑎) is unknown, so it is estimated as 𝑄𝑡(𝑎)  
Ideally, 𝑄𝑡(𝑎) should be as close as possible to 𝑞∗(𝑎) 
The agent must also balance exploration (learning about 
actions) and exploitation (using the best-known action) 
45 


## Page 46

The Exploration-Exploitation Dilemma 
Exploitation: choosing the action with the highest estimated 
value 𝑄𝑡(𝑎) 
Maximizes short-term gain (greedy) 
May get stuck in suboptimal actions 
Exploration: trying less-known actions to improve knowledge 
Helps find better options in the long run 
May lead to temporary lower rewards 
46 


## Page 47

Action Selection Strategies 
Greedy strategy 
Always select 𝐴𝑡= argmax
𝑎
𝑄𝑡(𝑎) 
Maximizes immediate reward, but will never discover better actions 
є-greedy strategy 
With probability 1 − є, select argmax
𝑎
𝑄𝑡(𝑎) 
With probability є, pick a random action 
Ensures ongoing exploration while still favoring high-value 
actions 
47 


## Page 48

Sample-Average Estimation of Action Values 
The sample average method updates action value estimates as: 
 
 
 
The indicator function 𝟙 returns 1 if a condition is true, otherwise it returns 0 
If an action is selected many times, Qt(a) converges to q∗(a), according 
to the law of large numbers 
This method is unbiased (produces estimates that are correct on 
average), but slow to adapt in changing environments 
It works best when the reward distribution is stationary 
48 


## Page 49

Incremental Implementation 
49 
the error in the estimate 
Here, n is the 
number of times 
the action has been 
selected up to the 
current time step 


## Page 50

Incremental Implementation 
Advantages 
Requires less memory: constant memory, does not store all rewards 
Computationally efficient: constant-time updates per step 
Disadvantages 
Adapts slowly in nonstationary environments 
Better alternatives exist for dynamic problems, e.g., weighted 
updates (next subsection) 
 
50 


## Page 51

Pseudocode 
51 


## Page 52

Example: A 10-Armed Bandit 
The true value q*(a) of each of the 10 actions was selected according to a normal distribution 
with mean 0 and variance 1, and then the actual rewards were selected according to a normal 
distribution with mean q*(a) and variance 1 


## Page 53

Experiment 1 
1 Simple Bandit.py 
 
The 10-armed testbed 
2000 randomly generated k-armed bandit problems, with k = 10 
actions 
Each action value is chosen randomly from a normal distribution 
The agent’s performance is measured over 10 000 time steps 
53 


## Page 54

Results: Average Reward 
54 


## Page 55

Results: % Optimal Action 
55 


## Page 56

Greedy vs. є-Greedy 
The greedy method gets stuck and reaches a lower average reward  
of ~1 
It selects the optimal action in only ~1/3 of the cases 
є-greedy with є = 0.1 explores more and finds the best action faster 
є-greedy with є = 0.01 explores slower, but achieves better long-term 
performance than є = 0.1 
56 


## Page 57

Greedy vs. є-Greedy 
If reward variance is 0, greedy can perform well by identifying the 
best action in one trial and exploiting it from then on 
When reward variance is high, є-greedy performs better than greedy 
since more exploration is needed to estimate action values 
The assumption of stationarity often breaks in practice; action values 
can change due to environment modifications or changes in behavior 
caused by learning 
In nonstationary tasks, continual exploration is necessary to detect 
when previously suboptimal actions become better than the current 
greedy choice 
57 


## Page 58

Asymptotic Guarantees of є-Greedy 
For a (theoretically) infinite number of steps, the є-greedy 
method guarantees that: 
Every action will be sampled an infinite number of times  
 ⇒ 𝑄𝑡(𝑎) →𝑞∗(𝑎) for all actions 
The probability of selecting the optimal action exceeds 1 – є 
(approaches certaintly) 
58 


## Page 59

Introduction. Multi-Armed Bandits 
1. Introduction to Reinforcement Learning 
 
1.1. RL Among Machine Learning Paradigms 
 
1.2. Key Elements of RL 
 
1.3. Related Fields 
2. Multi-Armed Bandit Problems 
 
2.1. Sample-Average Estimation of Action Values 
 
2.2. Exponential Recency-Weighted Average 
 
2.3. Optimistic Initial Estimates 
 
2.4. Upper-Confidence-Bound Action Selection 
 
 
59 


## Page 60

Nonstationary Problems 
Incremental updates avoid the need to store all past rewards 
 
 
For nonstationary environments (where reward distributions change 
over time), this update rule is changed with a constant step-size 
parameter α ∊ (0, 1]  
 
 
This is called exponential recency-weighted average, because over 
time, the weight of past rewards decays exponentially 
This makes recent rewards more influential, which is essential in 
nonstationary settings 
 
𝑄𝑛+1  = 𝑄𝑛 + 𝛼𝑅𝑛−𝑄𝑛 
60 


## Page 61

Exponential Recency-Weighted Average 
61 

(1 −𝛼)𝑛−𝑖 is the weight given to Ri and it decreases with the number of 
intervening  time steps; older rewards count less 

Example: assume α = 0.1 and n = 100 steps 

The weight of step 99 is: 0.1 ∙1 −0.1 100−99 = 0.1 ∙0.91 ≈10−1 

The weight of step 20 is: 0.1 ∙1 −0.1 100−20 = 0.1 ∙0.980 ≈2 ∙10−5 
 


## Page 62

Exponential Recency-Weighted Average 
62 
Larger α values make updates more responsive to recent changes 
Smaller α values make estimates more stable but slower to adapt 
Adaptive strategies can dynamically adjust α over time 


## Page 63

Sample Average vs.  
Exponential Recency-Weighted Average  
Sample average 
Weighs all past rewards equally 
Is slow to adapt when action values change 
Works well for stationary environments 
Exponential recency-weighted average  
Prioritizes recent data over older observations 
Adapts quickly to changing rewards 
Is ideal for nonstationary environments 
 
ERWA with 𝛼𝑛𝑎= 1
𝑛  is equivalent to SA 
63 


## Page 64

Convergence 
Convergence is guaranteed if: 
 
 
The first condition guarantees that the steps are large enough 
to eventually overcome any initial bias or random fluctuations 
The second condition guarantees that eventually the steps 
become small enough to assure convergence  
Suitable functions are: 
𝛼𝑛𝑎= 1
𝑛  
𝛼𝑛𝑎=
1
𝑛+1   ∈(0, 1]  
64 


## Page 65

Convergence 
For constant step size, i.e., 𝛼𝑛𝑎= 𝛼, convergence is not 
guaranteed 
In practice: 
Adaptive step size sequences may converge very slowly or require 
fine-tuning 
Constant step size methods often perform well enough, even in 
nonstationary environments 
The trade-off between exploration and exploitation remains 
important 
65 


## Page 66

Experiment: Comparing Methods 
An experiment with the 10-armed testbed where the true 
values of the actions change over time 
Uses sample averages vs. a constant step-size method with  
α = 0.1 and є = 0.1 
The experiment is run for 10 000 steps and performance is 
compared  
66 


## Page 67

Experiment 2 
2 Simple Bandit Nonstationary.py 
 
q*(a) start out equal and then take independent random walks 
by adding a normally distributed increment with mean 0 
and standard deviation 0.01 to all the q*(a) on each step  
 
67 


## Page 68

Results: Average Reward 
68 


## Page 69

Results: % Optimal Action 
69 


## Page 70

Introduction. Multi-Armed Bandits 
1. Introduction to Reinforcement Learning 
 
1.1. RL Among Machine Learning Paradigms 
 
1.2. Key Elements of RL 
 
1.3. Related Fields 
2. Multi-Armed Bandit Problems 
 
2.1. Sample-Average Estimation of Action Values 
 
2.2. Exponential Recency-Weighted Average 
 
2.3. Optimistic Initial Estimates 
 
2.4. Upper-Confidence-Bound Action Selection 
 
 
70 


## Page 71

Initial Bias: Example 
Initial action value estimates can influence exploration 
Action A: true value = 0.5 
Action B: true value = 0.8 (best action) 
Suppose the rewards are noisy and vary around the true value. For 
simplicity, assume small noise so that rewards are close to the true 
value 
The agent uses a greedy strategy and updates with sample averages 
71 


## Page 72

Neutral Initial Estimates 
Initial estimates: 

Q(A) = 0 

Q(B) = 0 
Assume the agent picks Action A first and gets a reward around 0.5 
Updates: 

Q(A) = 0.5 

Q(B) = 0 
Now the greedy policy chooses Action A again. More rewards come in 
around 0.5. Q(A) stays near 0.5. B is never tried 
The agent may never discover that Action B is better. It is stuck 
exploiting a suboptimal action 
 
72 


## Page 73

Optimistic Initial Estimates 
Initial estimates: Q(A) = 5, Q(B) = 5 
Suppose the agent picks Action A first and gets a reward around 0.5 
Updates: Q(A) = ~0.5, Q(B) = 5 

Q(A) = 5 is only an initial estimate, not a pseudo-sample. The value 5 is not 
considered in the average of real samples 
Now B looks better. The agent chooses Action B next and gets a 
reward around 0.8 
Updates: Q(A) = ~0.5, Q(B) = ~0.8 
Now the greedy strategy favors Action B, which is the optimal action 
73 


## Page 74

Encouraging Exploration with Optimistic 
Values 
The optimistic initial values method sets initial estimates higher to 
encourage exploration: Q1(a) = Qhigh  
The agent begins by selecting actions with high initial estimates 
When an action is chosen, the received reward is lower than expected  
This forces the agent to try all actions, since they all appear 
promising initially  
Unlike є-greedy, exploration is systematic rather than random 
74 


## Page 75

Optimistic Initialization vs. є-Greedy 
Optimistic initialization explores early, but stops exploring once values 
stabilize 
є-greedy continues to explore throughout the learning process 
Optimistic initialization 
Initially performs worse due to exploration, but eventually performs better as 
exploration decreases 
Works well if the best action does not change 
Converges faster in stationary settings 
Works well on stationary problems, but may fail with nonstationary tasks 
є-greedy 
Adapts to nonstationary environments 
Requires tuning є to balance exploration and exploitation 
75 


## Page 76

Experiment 3 
3 Simple Bandit Optimistic Start.py 
 
Optimistic initial values (OIV): Q1(a) = +5 a, greedy 
Comparison with є-greedy with є = 0.1 
76 


## Page 77

Results: Average Reward 
77 
large oscillations 


## Page 78

Results: % Optimal Action 
78 
large oscillations 


## Page 79

Results 
Initially, OIV performs worse due to greater exploration, but over 
time it converges to higher reward levels as exploration drives better 
estimates 
OIV does a lot of forced exploration early, due to optimism 
The effect of randomness in early rewards is magnified, because only 
one sample is used at first 
As a result, the early performance is unstable, even when averaged 
over many runs 
79 


## Page 80

Introduction. Multi-Armed Bandits 
1. Introduction to Reinforcement Learning 
 
1.1. RL Among Machine Learning Paradigms 
 
1.2. Key Elements of RL 
 
1.3. Related Fields 
2. Multi-Armed Bandit Problems 
 
2.1. Sample-Average Estimation of Action Values 
 
2.2. Exponential Recency-Weighted Average 
 
2.3. Optimistic Initial Estimates 
 
2.4. Upper-Confidence-Bound Action Selection 
 
 
80 


## Page 81

Upper Confidence Bound (UCB) Action 
Selection 

Instead of random exploration, actions could be selected according to their 
potential for being optimal 

How close their estimates are to being maximal  

The uncertainties in those estimates 

Upper Confidence Bound (UCB) selection rule: 
 
 
 

Qt(a) is the exploitation part 

ln 𝑡
𝑁𝑡(𝑎) is the exploration part (bonus for less-visited actions) 

c controls the degree of exploration; usually, c = 2 

If Nt(a) = 0, then a is considered to be a maximizing action  
 
81 
ln
argmax
( )
( )
t
t
a
t
t
A
Q a
c
N a










Nt(a) is the number 
of times that action a 
has been selected 
before time step t 


## Page 82

UCB Action Selection 
The square root term in UCB reflects the uncertainty in the estimated 
value of action a 
The UCB formula sets an upper bound on the possible true value of 
action a 
Selecting action a reduces uncertainty by increasing Nt(a) 
Not selecting a increases t but not Nt(a), which raises the uncertainty 
term 
The logarithmic term grows slowly, and this ensures eventual 
selection of all actions 
Actions with low values or high selection counts are chosen less 
frequently over time 
82 


## Page 83

UCB vs. є-Greedy 
UCB prioritizes actions with high uncertainty, unlike є-greedy 
which selects randomly 
Exploration reduces naturally over time, which ensures 
efficient long-term learning 
Advantages:  
No need to tune є 
Better theoretical guarantees 
Disadvantages:  
Assumes a stationary environment 
More complex to implement than є-greedy 
 
83 


## Page 84

Experiment 4 
4 Simple Bandit UCB.py 
84 


## Page 85

Results: Average Reward 
85 
Similar results  
for є = 0.01 


## Page 86

Results: % Optimal Action 
86 
Similar results  
for є = 0.01 


## Page 87

Results 
UCB generally achieves higher rewards over time 
Early exploration is more structured, and leads to faster 
convergence 
UCB often outperforms є-greedy except in nonstationary cases 
If the optimal action changes over time, UCB may get stuck due to 
insufficient exploration 
Hybrid approaches like UCB with weighted averages may perform 
better in such cases (Qt , t and Nt are weighted with discount factors) 
87 


## Page 88

Conclusions 
Reinforcement learning is about agents learning to make 
decisions through interaction and delayed reward 
The agent-environment framework defines how actions, states, 
and rewards influence learning over time 
Bandit problems are a simplified form of RL, focused only on 
action selection and reward estimation, without states 
Exploration strategies like є-greedy, optimistic values, and UCB 
are important to avoid getting stuck on suboptimal actions 
88 


## Page 89

Main References 
Sutton, R.S. and Barto, A.G. (2018). Reinforcement Learning: An 
Introduction. 2nd edition. MIT Press, Cambridge, MA. 
http://incompleteideas.net/book/the-book-2nd.html 
Plaat, A. (2022). Deep Reinforcement Learning, Springer. 
https://arxiv.org/pdf/2201.02135 
 
89 
