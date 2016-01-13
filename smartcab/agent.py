import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import defaultdict
import operator
import numpy as np
from random import choice

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.env = env
        self.state = None
        self.next_waypoint = None
        self.counter_cw = np.array([[0,1],[-1,0]])
        self.clockwise = np.array([[0, -1],[1,0]])
        self.gamma = 0.5
        self.qhat = None
        self.episilon = None
        self.old_location = []
        self.old_action = None

        # TODO: Initialize any additional variables here

    def reset(self, destination=None):
        self.qhat = defaultdict(lambda: dict((e, 0) for e in self.env.valid_actions))
        self.planner.route_to(destination)
        self.step = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state = self.env.agent_states[self]
        location = np.array(self.state['location'])
        heading = np.array(self.state['heading'])
        self.step += 1

        # sort a dict of actions by maximum utility descending
        sorter = lambda d : sorted(d.items(), key=operator.itemgetter(1), reverse=True)

        # given a dict of action/utility pairs, return a list of all actions with equal utility
        dups = lambda d, u: [k for k, v in d.iteritems() if v == u]

        # calculate next locations by adding the current location coordinates to rotated headings
        new_locations = {'left': tuple(location + heading.dot(self.counter_cw)),
                         'right': tuple(location + heading.dot(self.clockwise)),
                         'forward': tuple(location + heading)
                        }

        # Policy is to select action based on estimated maximum utility of the next state

        # dict of sorted current Q(a',s') values at next locations
        scores = {'left': sorter(self.qhat[new_locations['left']])[0][1],
                  'right': sorter(self.qhat[new_locations['right']])[0][1],
                  'forward': sorter(self.qhat[new_locations['forward']])[0][1]
                 }

        if list(location) == self.old_location:
            print self.old_action
            print inputs

        # calculate the learning rate for each step so that actions are random initially and
        # the learning rate increases with time.
        alpha_t = 1./self.step**2

        if random.random() > 1 - alpha_t:
            # choose a completely random action if draw from a uniform distribution is > 1 - alpha_t
            action = random.choice([e for e in self.env.valid_actions if e is not None])
        else:
            # choose from all actions with equal maximum utility randomly
            action = choice(dups(scores, sorter(scores)[0][1]))

        self.old_action = action
        self.old_location = list(location)

        # avoid 'None' action by turning right on red
        if inputs['light'] == 'red' and inputs['left'] == None:
            action = 'right'

        # Execute action and get reward
        reward = self.env.act(self, action)

        # get scores at new location
        scores = self.qhat[new_locations[action]]

        # find utility of next state
        qhat = max(scores.iteritems(), key=operator.itemgetter)[0]

        # calculate discounted utility of current state
        self.qhat[tuple(location)][action] = reward + self.gamma * scores[qhat]


        if self.state['location'] == self.state['destination']:
            print 'Destination found in %d steps.\n' %self.step
        # else:
        #     print "location {} rewards {}".format(self.state['location'], self.qhat[self.state['location']], )

        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=1.0)  # reduce update_delay to speed up simulation
    sim.run(n_trials=10)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
