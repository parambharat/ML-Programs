#P4: Train a smart cab to drive 
##Implement a basic driving agent

<p>
A basic driving agent, which processes the following inputs at each time step:
<ul>
<li>Next waypoint location, relative to its current location and heading</li>
<li>Intersection state (traffic light and presence of cars)</li>
<li>Current deadline value (time steps remaining)</li>
</p>
<p>
Run this agent within the simulation environment with enforce_deadline set to False (see run function in agent.py), and observe how it performs. In this mode, the agent is given unlimited time to reach the destination. The current state, action taken by your agent and reward/penalty earned are shown in the simulator.
</p>
<p>**In your report, mention what you see in the agent's behavior. Does it eventually make it to the target location?**</p>

<p>
The agent makes it to the target location even when the actions are chosen at random. However, it's arguably not the desired behavior from a smartcab. 
It must be able to plan and move in an optimal path from the source to the destination. Choosing a random action suggests that there is no learning and hence nothing smart about the cab.
</p>

##Identify and update state

<p>**Justify why you picked these set of states, and how they model the agent and its environment.**</p>