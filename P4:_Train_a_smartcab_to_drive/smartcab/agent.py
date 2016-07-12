import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import itertools as it
import pandas as pd
class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.alpha = 0.7     # learning_rate
        self.gamma = 0.2    # discount_factor
        self.epsilon = 0.1  # random likelihood for GLIE 
        self.net_reward = 0
        self.net_penalty = 0
        self.qTable = self.getInitialQvalues()

    def getInitialQvalues(self):
        # Create a qtable as a cross product of all states and rewards
        self.actions = Environment.valid_actions
        # possible states values for Q_table
        lights,waypoint,oncoming,left,right =(['red','green'],self.actions,
                                              self.actions,self.actions,
                                              self.actions)
        all_states = it.product(lights,waypoint,oncoming,left,right)
        return pd.DataFrame(columns=self.actions,index=all_states).fillna(1)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.net_reward = 0
        self.net_penalty = 0

    def getState(self,inputs):
        # a representation of state.
        state =(inputs['light'],inputs['waypoint'], inputs['oncoming'],
                inputs['left'], inputs['right'])
        # excluding deadline from the state. 
        return state

    def getAction(self,state):
        # Get the next action from qTable
        actions = self.qTable.loc[[state]]
        return actions.idxmax(axis=1)[0]

    def getQvalue(self,state,action):
        return self.qTable.loc[[state]][action][0]

    def getUpdatedQ(self,action,reward):
        n_waypoint = self.planner.next_waypoint()
        n_inputs = self.env.sense(self)
        n_inputs['waypoint'] = n_waypoint
        n_state = self.getState(n_inputs)
        n_action = self.getAction(n_state)
        n_qvalue = self.getQvalue(n_state,n_action)
        utility = reward + self.gamma*n_qvalue
        qvalue = self.getQvalue(self.state,action)
        return (1 - self.alpha)*qvalue + (self.alpha*utility)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        inputs['waypoint'] = self.next_waypoint
        inputs['deadline'] = deadline

        # TODO: Update state
        self.state = self.getState(inputs) # get the current state.

        # TODO: Select action according to your policy
        #action = random.choice(self.actions) #selecting a random action
        action = self.getAction(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward>0:
            self.net_reward +=reward
        else:
            self.net_penalty +=reward

        # TODO: Learn policy based on state, action, reward
        print self.getUpdatedQ(action,reward)
        self.qTable.set_value(self.state, action, self.getUpdatedQ(action,reward))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.02, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
