
# P4: Train a smart cab to drive 

## Implement a basic driving agent

A basic driving agent, which processes the following inputs at each time step:
<ul>
<li>Next waypoint location, relative to its current location and heading</li>
<li>Intersection state (traffic light and presence of cars)</li>
<li>Current deadline value (time steps remaining)</li>
</ul>

<p>Run this agent within the simulation environment with enforce_deadline set to False (see run function in agent.py), and observe how it performs. In this mode, the agent is given unlimited time to reach the destination. The current state, action taken by your agent and reward/penalty earned are shown in the simulator.
</p>

### In your report, mention what you see in the agent's behavior. Does it eventually make it to the target location?

<p>
The agent eventually makes it to the target location quite a few times when the actions are chosen at random. However, it also misses it's target location and hits a hard stop a few times. It is purely chance that the agent reaches it's destination and often does so way past the deadline. It's equally likely that it doesn't reach the deadline either.
</p>

Further, the following was noticed here with respect to the agent, environment and rewards:
1. <p>The agent had no regard for traffic rules, lights, right-of-way, waypoint and sense of direction.</p>
2. <p>It seems to be penalized for the following actions as seen from the results:</p>
    <p>a. `-0.5` when, the action taken doesn't lead to the next waypoint.<br>
    Ex: `{'tripId': 1, 'right': None, 'light': 'green', 'oncoming': None, 'deadline': 23, 'waypoint': 'forward', 'action': 'right', 'reward': -0.5, 'left': None}`</p>
OR<br>
    <p> when, the light is red and an action to turn right is executed provided there is no traffic at the waypoint.<br> Ex: `{'right': None, 'tripId': 2, 'light': 'red', 'oncoming': None, 'deadline': 12, 'waypoint': 'forward', 'action': 'right', 'reward': -0.5, 'left': None}`</p>
    
    <p>b. `-1` When, the action taken doesn't obey the traffic signal or when it takes an action against the U.S. right-of-way.<br> Ex: `{'tripId': 100, 'right': None, 'light': 'red', 'oncoming': None, 'deadline': -64, 'waypoint': 'right', 'action': 'forward', 'reward': -1.0, 'left': None}`</p>

3. <p>No reward is given when no action is taken by the agent while it could have instead.<br> Ex: `{'action': nan, 'deadline': -50, 'left': None, 'light': 'green', 'oncoming': None, 'reward': 0.0, 'right': None, 'tripId': 100, 'waypoint': 'right'}`</p>

4. <p>A positive reward is given when the agent executes  actions that takes it to the `next_waypoint` without disrupting traffic.<br> Ex: `{'action': 'right', 'deadline': -83, 'left': None, 'light': 'green', 'oncoming': None, 'reward': 2.0, 'right': None, 'tripId': 7, 'waypoint': 'right'}`</p>

<p>However, this is arguably not the desired behavior expected from a smartcab. It must be able to plan and move in an optimal path from the source to the destination. Choosing a random action suggests that there is no learning and hence nothing smart about the cab.
</p>

## Identify and update state

<p>Identify a set of states that you think are appropriate for modeling the driving agent. The main source of state variables are current inputs, but not all of them may be worth representing. Also, you can choose to explicitly define states, or use some combination (vector) of inputs as an implicit state.</p>

<p>At each time step, process the inputs and update the current state. Run it again (and as often as you need) to observe how the reported state changes through the run.</p>

### Justify why you picked these set of states, and how they model the agent and its environment

<p>A state is representative of the condition that the agent is in with respect to the environment at any particular given point in time. It seems appropriate to represent it as a combination of sensory inputs and the responses derived from the environment. Hence, we may choose to represent the state with the following inputs:</p>

1. `Light: ['Red','Green']`
2. `Waypoint: ['None','Left','Right','Forward']`
3. `Left: ['None','Left','Right','Forward']`
4. `Oncoming: ['None','Left','Right','Forward']`
5. `Right: ['None','Left','Right','Forward']`
6. `deadline: Number of points in the Grid`

<p>`Light`, `action` and `oncoming` captures information about traffic rules in the environment and the `waypoint`guides the agent to the next waypoint that will lead it to the destination.</p>

<p>Adding the `deadline` increases the state space and we wish to keep the state space as small as possible. Further, it's felt that since the `waypoint` is already guiding the agent toward the deadline already `deadline` seems redundant and unnecessary.</p>

<p>The final state space is now composed of `light`, `waypoint`, `left`, `oncoming` and `right` resulting in `2*4*4*4*4 = 512` possible states. </p>

## Implement Q-Learning
<p>Implement the Q-Learning algorithm by initializing and updating a table/mapping of Q-values at each time step. Now, instead of randomly selecting an action, pick the best action available from the current state based on Q-values, and return that.</p>

<p>Each action generates a corresponding numeric reward or penalty (which may be zero). Your agent should take this into account when updating Q-values. Run it again, and observe the behavior.</p>

### What changes do you notice in the agent’s behavior?
<p>The agent reaches the destination more often and is significantly faster after Q-Learning is implemented. It's however, important to note that the agent begins to learn only as the number of trials increase. In the beginning *(first 10 trips)* although the agent reaches the destination on time it takes longer and collects more negative rewards to reach the destination while towards the end *(last 10 trips)* it begins to accumulate more positive rewards and reaches the destination much faster.</p>
<p>This shows that the agent learns from its mistakes progressively and learns the behavior that the environment rewards as the trips  progress. Further, it was noticed that the agent explores the environment in the beginning and begins to exploit as the trip count increases.<p>
<p>At this point the `learning rate` (&alpha;) and the `discount rate` (&gamma;) have been arbitarily chosen to be `0.5` and `0.5` respectively. Further, no greedy-exploration of the environment is implemented yet.</p>

## Enhance the driving agent
<p>Apply the reinforcement learning techniques you have learnt, and tweak the parameters (e.g. learning rate, discount factor, action selection method, etc.), to improve the performance of your agent. Your goal is to get it to a point so that within 100 trials, the agent is able to learn a feasible policy - i.e. reach the destination within the allotted time, with net reward remaining positive.</p>

### Report what changes you made to your basic implementation of Q-Learning to achieve the final version of the agent. How well does it perform?

<p>The action selection method is already quite optimal and the vannila Q-learning implementation proved to be quite effective. Toward the end of 100 trips it can be seen that the agent seldom collects negative feedback.</p>

<p>However, we could tinker with the `learning rate` (&alpha;) and the `discount rate` (&gamma;) to optimize this further. We can also implement `greedy limit infinite exploration` so that the agent explores the environment to ensure that it reaches the destination in the minimum possible time.</p>

<p> Further, the state space could be reduced. The car in the `right` although present adds no value with the current reward system in place. Further, in the U.S. right-of-way rules we are concerned with the car `oncoming` and the car on the `left`. Hence, it makes sense to ignore `right` while reducing the state space.</p>

<p> Since, we wish to change the hyperparameters of learning i.e. &alpha; , &gamma; and &epsilon; we must have a metric or measure to judge how well the agent is doing and must be able to compare the results using this metric/measure. While there could be multiple measures/metrics we could use, it was decided that we shall go with `MeanNetReward/Action`. Mathematically, this is: $$\frac{NetRewards}{Actionstaken}$$ Here,<br> `NetReward = Sum of all rewards in 100 trials` <br>`Actionstaken = Total number of actions taken in 100 trials`</p>
<p> Further, it also makes sense to track the number of times the agent reaches the destination during the 100 trials</p>

<p>These have been recorded in the tables below for all the three attempts:</p>
    
1. Random Actions picked
2. Vanilla implementation of Q-Learning.
3. Enhanced Q-Learning.

<style>
.table{
    padding:2px 3px 2px 3px;
    vertical-align:bottom;
    border-right:1px solid #000000;
    border-bottom:1px solid #000000;
    text-align:right;
    }
</style>
<table cellspacing="0" cellpadding="0" dir="ltr" border="2" style="table-layout:fixed;font-size:13px;font-family:arial,sans,sans-serif;border-collapse:collapse;border:1px solid #ccc">
  <caption>Random Actions</caption>
  <tbody>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Trial&quot;}">Trial</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;MeanReward/Step&quot;}">MeanReward/Action</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Reached&quot;}">Reached</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Not Reached&quot;}">Not Reached</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">1</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.03326023392</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">23</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">77</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">2</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.01367488444</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">19</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">81</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">3</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.05433130699</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">24</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">76</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">4</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.01201550388</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">22</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">78</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">5</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.01397168405</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">23</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">77</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">6</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.08922226609</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">30</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">70</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">7</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.008302583026</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">19</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">81</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">8</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">-0.004234297812</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">19</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">81</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">9</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.03193960511</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">22</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">78</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">10</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.03458049887</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">23</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">77</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">Mean</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">0.02870642686</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">22.4</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">77.6</td>
    </tr>
  </tbody>
</table>

<table cellspacing="0" cellpadding="0" dir="ltr" border="2" style="table-layout:fixed;font-size:13px;font-family:arial,sans,sans-serif;border-collapse:collapse;border:1px solid #ccc">
  <caption>Vanilla Q-Learning (learning_rate = 0.5, discount_factor=0.5)</caption>
  <tbody>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Trial&quot;}">Trial</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;MeanReward/Step&quot;}">MeanReward/Action</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Reached&quot;}">Reached</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Not Reached&quot;}">Not Reached</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">1</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.930419269</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">100</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">0</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">2</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.741129032</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">97</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">3</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">3</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.656502572</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">97</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">3</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">4</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.799159664</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">98</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">2</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">5</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.740930599</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">100</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">0</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">6</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.664661654</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">96</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">4</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">7</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.624905944</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">96</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">4</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">8</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.603606789</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">98</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">2</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">9</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.701638066</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">96</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">4</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">10</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.707354056</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">96</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">4</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">Mean</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.717030764</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">97.4</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">2.6</td>
    </tr>
  </tbody>
</table>

<table cellspacing="0" cellpadding="0" dir="ltr" border="2" style="table-layout:fixed;font-size:13px;font-family:arial,sans,sans-serif;border-collapse:collapse;border:1px solid #ccc">
  <caption>Enhanced Q-Learning (learning_rate = 1, discount_factor=0.1)</caption>
  <tbody>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Trial&quot;}">Trial</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;MeanReward/Step&quot;}">MeanReward/Action</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Reached&quot;}">Reached</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:2,&quot;2&quot;:&quot;Not Reached&quot;}">Not Reached</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">1</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.789624183</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">99</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">1</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">2</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">2.005597015</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">99</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">1</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">3</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.837073982</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">98</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">2</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">4</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.900690846</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">99</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">1</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">5</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.926165803</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">100</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">0</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">6</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.802474062</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">99</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">1</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">7</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.848229342</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">100</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">0</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">8</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.893435635</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">100</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">0</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">9</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.753949447</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">99</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">1</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">10</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.938958707</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">100</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">0</td>
    </tr>
    <tr>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:1}">Mean</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:0.0332602339181}">1.869619902</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:23}">99.2</td>
      <td class="table" data-sheets-value="{&quot;1&quot;:3,&quot;3&quot;:77}">0.8</td>
    </tr>
  </tbody>
</table>


### Does your agent get close to finding an optimal policy, i.e. reach the destination in the minimum possible time, and not incur any penalties?

<p>The enhanced agent is quite close to finding an optimal policy. It reaches the destination in very little time when compared to the vanilla Q-learning implementation. This is visible from the fact that on average of 10 sets of 100 trials the agent with an &alpha;` = 1` and &gamma;` = 0.1` has a better MeanRewards per Action performed and reaches the target on a average 99.2 times.</p>

<p>Furthermore, it was quite counter-intuitive to notice that implementing the Greedy limit - Infinite exploration actually reduced the performance of the Agent although &epsilon; was `0.1` and hence this was reverted back</p>

