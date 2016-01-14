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
    n_trials = 10

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.env = env
        self.state = None
        self.next_waypoint = None
        self.Q = None
        self.episilon = None
        self.within_100 = 0
        self.within_deadline = 0
        self.deadline = 0
        self.gamma = 0.001

        # rotation matrices
        self.counter_cw = np.array([[0,1], [-1,0]])
        self.clockwise = np.array([[0, -1], [1,0]])

    def reset(self, destination=None):
        self.Q = defaultdict(lambda: dict((e, 0) for e in self.env.valid_actions))
        self.planner.route_to(destination)
        self.deadline = self.env.agent_states[self]['deadline']
        self.step = 0

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        self.state = self.env.agent_states[self]
        location = np.array(self.state['location'])
        heading = np.array(self.state['heading'])
        max_reward = 2
        self.step += 1

        # lambda to sort a dict of actions by maximum utility descending
        sorter = lambda d : sorted(d.items(), key=operator.itemgetter(1), reverse=True)

        # lambda to return a list of all actions with equal utility given a dict of action/utility pairs
        dups = lambda d, u: [k for k, v in d.iteritems() if v == u]

        # calculate next step locations by adding the current location coordinates to rotated headings
        new_locations = {'left': tuple(location + heading.dot(self.counter_cw)),
                         'right': tuple(location + heading.dot(self.clockwise)),
                         'forward': tuple(location + heading)}

        # dict of sorted current Q(a',s') values at next locations
        Q_sorted = {e: sorter(self.Q[new_locations[e]])[0][1]
                    for e in self.env.valid_actions if e is not None}

        # Policy is to select action based on historical maximum utility of the next state.
        # The learning rate alpha_t is used to randomize the rewards choice initially until
        # sufficient "exploration" has taken place. The rewards for all outcomes
        # are stored in the Q matrix. After the explotration 'phase', the 'exploitation'
        # phase determines the choice by comparing the highest possible reward outcome for
        # the next state in the Q matrix. If this reward is less than the maximum reward
        # then the action taken is the next waypoint.

        # calculate the learning rate for each step so that actions are random initially and
        # the learning rate increases with time.
        epsilon = 1./self.step**4
        alpha_t = 1 - epsilon

        # choose a completely random action if draw from a uniform distribution is > 1 - alpha_t
        if random.random() > alpha_t:
            action = random.choice([e for e in self.env.valid_actions if e is not None])
        else:
            # choose from all actions with equal maximum utility randomly
            if sorter(Q_sorted)[0][1] > max_reward:
                action = choice(dups(Q_sorted, sorter(Q_sorted)[0][1]))
            # choose the next waypoint
            else:
                action = self.next_waypoint

        # Execute action and get reward
        reward = self.env.act(self, action)

        # get scores at new location
        Q_sorted = self.Q[new_locations[action]]

        # find utility of next state
        qhat = max(Q_sorted.iteritems(), key=operator.itemgetter)[0]

        # calculate discounted utility of current state
        self.Q[tuple(location)][action] = reward + self.gamma * Q_sorted[qhat]

        if self.state['location'] == self.state['destination']:
            if self.step <= self.deadline:
                self.within_deadline += 1
            if self.step < 100:
                self.within_100 += 1
            print 'This trial destination found in %d steps.\n' %self.step
            print 'Destinations found within 100 steps : %d/%d' %(self.within_100, LearningAgent.n_trials)
            print 'Destinations found within deadline : %d/%d\n' %(self.within_deadline, LearningAgent.n_trials)

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0)  # reduce update_delay to speed up simulation
    sim.run(LearningAgent.n_trials)  # press Esc or close pygame window to quit


if __name__ == '__main__':
    run()
