# Novice Serpentine Coding Challenge
Well hi there! Welcome to the novice level Serpentine Coding Challenge. 
This challenge is meant to give an impression as to the sort of activities you could expect to be doing
as a coding member of the AI E-Sports team Serpentine! The novice level is meant for people with little 
programming experience. If you're up for more of a challenge, check out the Apprentice and Expert variants!

First things first. This challenge is heavily based on [this article](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/).
by Satwik Kansal and Brendan Martin. You can use it for reference on the learning algorithm we'll be using
and as a reference for code. It also shows how the game that we'll be automating works. You will be working in a 
code template, so you have some functions already available for use, called _novice_taxigame.py_.

All the answers are available as well in __taxigame.py__! However, it's much more fun to work on this yourself 
and see how far you can get, right ;)?

## The Challenge - Taking Code For A Ride
Today we will take a look at a simple implementation of a reinforcement learning algorithm: Q-learning.
Using this algorithm we can solve a basic game available in OpenAI Gym, called Taxi. The rules are simple:
![Taxi game](https://www.learndatasci.com/media/images/Reinforcement_Learning_Taxi_Env.width-1200.png)
* The taxi has 6 moves available to it: "South", "North", "East", "West", "Pickup" and "Dropoff".
* The taxi has to go to a pickup point (marked in purple) to pick up the passenger in that space. It then has to move to 
the drop off point (marked in blue), and drop off the passenger. There are 4 points in total where a passenger 
can be picked up or dropped off which are marked by letters.
* The taxi can not move through walls and is not able to move off of the grid.
* Each correct drop off rewards 20 points. Each time step reduces 1 point. Each wrong pickup or drop off 
reduces the reward by 10 points.

The state space of our game can be calculated by 
__5 x 5 x 5 x 4 = 500__ total possible states (x position, y position, pickup points + inside taxi, drop off points).
Printing the game using _OpenAI Gym_ in _Python 3.6_ gives us something like the following figure:

```
+---------+
|R: | : :G|
| : : : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+ 

Action Space Discrete(6)
State Space Discrete(500)
```

Our challenge today will be to use a learning algorithm to teach the taxi how to best play the game. Let's get cracking!

## A Picture Is Worth A Thousand Words
I can hear you think: "Playing a game is nice and all, but I haven't even seen it yet! 
How am I supposed to show you my genius without a canvas?". 
Never fear, you can use the template called _novice_taxigame.py_ to get a feeling for the code.
If you open it up, you should see a host of imports and functions. Down below, it should look something like this:

```python
# Initialization of environment and necessary variables
env = gym.make("Taxi-v2").env
env.reset()

# Import a pre-trained q_table
q_tab = np.zeros([env.observation_space.n, env.action_space.n])
q_tab = load_q_table()

# Visually play a game of Taxi in the console
play_game(q_tab, env)

# Show average performance over 100 games
get_performance(q_tab, env)
```

Let's make sense of what we see here. First we use OpenAI Gym to make a new instance of the "Taxi-v2" game 
and store it in _env_. This way, we do not need to program the game ourselves! Resetting the so called environment 
is just good practice. Next we initialize a Q-table. We'll get into the theory later, but in short this Q-table is 
the thing we are "training". The values in the table determine the actions our taxi will take. For now we are loading
a pre-trained Q-table from a file called _q_table.pickle_ to get a sense of what we are working with!

Using both the Q-table and the game environment we actually plot a game. Using _play_game()_ one random game will be 
displayed in the console. Random here means: A random assignment of the passenger pickup and drop off point. Lastly 
it would be nice to get a sense of the performance of this particular Q-table. This can be done with _get_performance()_
and looks something like this:

```python
Results after 100 episodes:
Average timesteps per episode: 12.73
Average penalties per episode: 0.0
Average reward per move: 0.6496465043205027
```
The results are the average of 100 random games (episodes).
* __Average timestep__: Each move in the game counts as one timestep. 
A lower value means the agent plays the game more efficiently!
* __Average penalties__: Penalties are counted each time the Taxi attempts a non-valid drop off.
* __Average reward per move__: A higher value means the agent plays the game more efficiently, meaning 
faster and with less mistakes. 

Try to run the code! You should see similar results appear in your console :).

## Ok Nice! But How Do I Make My Taxi Learn?
Well you could use several different algorithms: Neural networks, Support Vector Machines or Decision Trees for example.
However for the purpose of teaching you the concepts of OpenAI Gym and learning algorithms today we will use _Q-Learning_.

When coding your best friend is Google, so definitely feel free to find out how things work by yourself :)!
Let me just give you a rundown of some theory to get you on your way though.

### Reinforcement Learning
Q-learning, which is a type of reinforcement learning, can be explained very intuitively by thinking about the 
classic pavlovian learning model. Think about it like this: You would like to discourage wrong actions and encourage
(reinforce) actions that serve your purpose. To do this you could use a reward! For example: Every time you get distracted
from work, you could agree with your friends that you buy coffee for them. However, every time you finish your work effectively,
you get a cookie! Goodbye Facebook, goodbye internet cat videos, you'll be productive in no time!

The same goes for our Taxi. Each time step that, we count a little negative reward of -1 to discourage it from waiting 
around and doing nothing. Each time it wrongly drops a passenger we severely punish it by enforcing a -10 reward. Wouldn't
want our passengers laying on the middle of the road now would we?
Dropping off the passenger gives a +20 reward. Using both the "carrot and the stick", rewarding and punishing, we try
to get the taxi to efficiently move passengers.

### So, You Mentioned Q-Learning?
Yes! Now that we have a basic grasp on the concept we'll be using we can actually try to implement it! 
Q-learning is simply a simple way to approach the learning challenge mathematically and programmatically.
First Things first, remember when we figured out that the number of states in the game is _500_, and the number
of actions the taxi can take is _6_? We can check the _observation space_ and _action space_ with some simple 
lines of code.

```python
print("Action Space {}".format(env.action_space))
print("State Space {}".format(env.observation_space))
```

An example of a state and the output would then be:

```commandline
+---------+
|R: | : :G|
| : : : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+ 

Action Space Discrete(6)
State Space Discrete(500)
```

Thinking about it logically, we would like to know what the most optimal action in each state would be considering 
our game. Optimal here means the action that maximizes our reward in that particular game. 
Why not make a matrix out of this? When initialized it would probably look something like this:

<html>
<table class="tg">
  <tr>
    <th class="tg-xxl6" colspan="2" rowspan="2"><span style="font-weight:bold">Q-Table</span></th>
    <th class="tg-wvni" colspan="6">Actions</th>
  </tr>
  <tr>
    <td class="tg-0pky">South(0)</td>
    <td class="tg-0pky">North(1)</td>
    <td class="tg-0pky">East(2)</td>
    <td class="tg-0pky">West (3)</td>
    <td class="tg-0pky">Pickup (4)</td>
    <td class="tg-0pky">Dropoff (5)</td>
  </tr>
  <tr>
    <td class="tg-3t3n" rowspan="3">Observations</td>
    <td class="tg-0pky">(0)</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
  </tr>
  <tr>
    <td class="tg-0pky">(...)</td>
    <td class="tg-0pky">⋮<br></td>
    <td class="tg-0pky">⋮</td>
    <td class="tg-0pky">⋮</td>
    <td class="tg-0pky">⋮</td>
    <td class="tg-0pky">⋮</td>
    <td class="tg-0pky">⋮</td>
  </tr>
  <tr>
    <td class="tg-0pky">(500)</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
  </tr>
</table>
</html>

And that is exactly what a Q-table is. By updating the values in the Q-table we can "train" the taxi to choose the best
move in a state and try to get maximum reward! After training the values it would look a little bit differently.

<table class="tg">
  <tr>
    <th class="tg-xxl6" colspan="2" rowspan="2"><span style="font-weight:bold">Q-Table</span></th>
    <th class="tg-wvni" colspan="6">Actions</th>
  </tr>
  <tr>
    <td class="tg-0pky">South(0)</td>
    <td class="tg-0pky">North(1)</td>
    <td class="tg-0pky">East(2)</td>
    <td class="tg-0pky">West (3)</td>
    <td class="tg-0pky">Pickup (4)</td>
    <td class="tg-0pky">Dropoff (5)</td>
  </tr>
  <tr>
    <td class="tg-3t3n" rowspan="3">Observations</td>
    <td class="tg-0pky">(0)</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
    <td class="tg-0pky">0</td>
  </tr>
  <tr>
    <td class="tg-0pky">(...)</td>
    <td class="tg-0pky">⋮<br></td>
    <td class="tg-0pky">⋮</td>
    <td class="tg-0pky">⋮</td>
    <td class="tg-0pky">⋮</td>
    <td class="tg-0pky">⋮</td>
    <td class="tg-0pky">⋮</td>
  </tr>
  <tr>
    <td class="tg-0pky">(500)</td>
    <td class="tg-0pky">5.221<br></td>
    <td class="tg-0pky">4.598</td>
    <td class="tg-0pky">-2.558<br></td>
    <td class="tg-0pky">-10.001</td>
    <td class="tg-0pky">-4.987</td>
    <td class="tg-0pky">5.898<br></td>
  </tr>
</table>

### Updating The Q-table
So to summarize what we need to do:
* Initialize a Q-table for all observation-action combinitions.
* Explore current state by choosing an actions in the current state (called S).
* Travel to the next state (S') by taking action (a).
* From all possible state (S'), select the one with the highest Q-value.
* Update the Q-table.
* Set the next state (S') as the current state (S).
* Continue until the goal state is reached.

The big question now of course becomes: How do we update the Q-table? 
Well we would like our taxi to try new moves every now and then to make sure it explores the state space and it fills all
the entries in the Q-table. To facilitate this we introduce a variable ϵ. It represents the rate of random actions our agent 
performs during learning.

Next we define the following formula to update Q-values 

```commandline
Q(S, a) <-- (1 - alpha) * Q(s,a) + alpha * (reward + gamma * max(Q(S', a)))
```
where
* α (alpha) is the learning rate (0<α≤1) - Just like in supervised learning settings, α is the extent to which our 
Q-values are being updated in every iteration.
* γ (gamma) is the discount factor (0≤γ≤1) - determines how much importance we want to give to future rewards. A high 
value for the discount factor (close to 1) captures the long-term effective award, whereas, a discount factor of 0 makes 
our agent consider only immediate reward, hence making it greedy.

Think about this equation for a bit and see if you understand why this updates the Q-values favourably.
If you get this step then all that is left is implementation! Let's get to the meat of the challenge!

## The Challenge!
Alright! Finally, some programming! Because this a introductory challenge for those with little coding experience we 
will take the functions already present and fill in some steps to get things to work. 

First, rewrite the bottom of your file to call the __train_taxi()__ function instead of the __load_q_table()__ function. 
Take note, because the first function takes two arguments: the Q-table and the environment, in order to work.

```python
 Initialization of environment and necessary variables
env = gym.make("Taxi-v2").env
env.reset()

# Import a pre-trained q_table
q_tab = np.zeros([env.observation_space.n, env.action_space.n])
q_tab = train_taxi(q_table, env)

# Visually play a game of Taxi in the console
# play_game(q_tab, env)

# Show average performance over 100 games
get_performance(q_tab, env)
```

Running this code will not train the taxi yet, that's your job ;)! You can play a game with an untrained taxi, and I 
encourage you to do just that by uncommenting __play_game()__. However you will see random behaviour. 
Think about why that is by thinking about how the Q-table right now. 

In order to make it learn, your job is to fill in two parts of __train_taxi()__. First the part about exploration:

```python
# Epsilon decides the rate of random actions
if random.uniform(0, 1) < epsilon:
    action = env.action_space.sample()  # Exploring action (random)
else:
    """Your code here! Remove the current code""" # Exploit learned action (q_table)
    action = env.action_space.sample()
```

And a part that updates the Q-value using the aforementioned equation: 

```python
 # Take a step in the taxi environment (e.g. do one of 6 actions)
next_state, reward, done, info = env.step(action)

# Calculate a new Q value based on the step made in the environment (learning)
"""

Your code here!

q_table[state, action] = ?

"""

# Some bookkeeping for next simulation step
if reward == -10:
    penalties += 1
state = next_state
epochs += 1
```

Good luck :)! Remember, solutions are available, but just have a go and see if you can understand what it all means.
It might be nice to try and get similar performance at the Q-table we loaded earlier. With the current Hyper Parameters 
(alpha, gamma, epsilon and the number of episodes set to 100000) it should be possible to train a similar table in under 45 seconds ;).

