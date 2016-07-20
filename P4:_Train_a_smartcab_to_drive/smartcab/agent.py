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
        self.alpha = 1     # initial learning_rate
        self.gamma = 0    # initial discount_factor
        self.epsilon = 0.1  # random likelihood for GLIE 
        self.tripId = -1
        self.qTable = self.getInitialQvalues()
        self.out = pd.DataFrame(columns=['tripId','deadline','inputs',
                                         'action','reward'])

    def getInitialQvalues(self):

        # Create a qtable as a cross product of all states and rewards
        self.actions = Environment.valid_actions

        # possible states values for Q_table

        lights,waypoint,oncoming,left = (['red','green'],self.actions,
                                              self.actions, self.actions)

        #cross-product of all relevant state variables

        all_states = it.product(lights,waypoint,oncoming,left)

        # create a dataframe to represent the qtable, index = states,
        # columns=actions and values = qvalues.
        return pd.DataFrame(columns=self.actions,index=all_states).fillna(1)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        self.tripId +=1

    def getState(self,inputs):
        # a representation of state.
        state =(inputs['light'],inputs['waypoint'], inputs['oncoming'],
                inputs['left'])
        # excluding deadline and right from the state. 
        return state

    def getAction(self,state):
        # Get the an action from qTable based on maximum qvalue
        actions = self.qTable.loc[[state]]
        return actions.idxmax(axis=1)[0]

    def getBestAction(self,state):
        p = random.random()
        decayed = (1 / ((self.tripId+1) ** (self.epsilon))) * self.epsilon
        if p<(self.epsilon):
            action = random.choice(self.env.valid_actions)
        else:
            action = self.getAction(state)
        return action

    def getQvalue(self,state,action):
        # Get the qvalue from the qtable based on state and action taken
        return self.qTable.loc[[state]][action][0]

    def getUpdatedQ(self,action,reward):
        # update the qtable with the qvalue based on value iteration of Q

        # Get the next waypoint from the new state
        n_waypoint = self.planner.next_waypoint()
        # Get the inputs for the new state
        n_inputs = self.env.sense(self)
        # assign waypoint as an input 
        n_inputs['waypoint'] = n_waypoint
        # The next state.
        n_state = self.getState(n_inputs)

        # The next action to take given the policy remains the same
        n_action = self.getAction(n_state)

        # The qvalue if the next action is taken
        n_qvalue = self.getQvalue(n_state,n_action)

        # The utility if the current policy is followed
        utility = reward + self.gamma*n_qvalue

        # The current qvalue for reaching the current state
        qvalue = self.getQvalue(self.state,action)

        # Updated qvalue for the current state based on value iteration 
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

        #reverting back to best action without randomness
        action = self.getAction(self.state)

        #actions with randomness for greedy exploration
        #action = self.getBestAction(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        self.qTable.set_value(self.state, action, self.getUpdatedQ(action,reward))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        self.out = self.out.append(pd.Series({"tripId":self.tripId,
                                              "deadline":deadline,
                                              "inputs":inputs,
                                              "action":action,
                                              "reward":reward}),
                                   ignore_index=True)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line

    a.out.to_csv('trial_data.csv',ignore_index=True)
if __name__ == '__main__':
    run()
