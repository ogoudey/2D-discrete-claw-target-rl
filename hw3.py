import numpy as np
from scipy.stats import mode

import random
from collections import defaultdict
import time
import traceback
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from tqdm import tqdm
import math
import copy

class Sim:

    def __init__(self, oml=0.0, robot_position=[0,0], object_position=[5,5], map_shape=(6,6)):
        self.robot_initial_position = robot_position
        self.object_initial_position = object_position
        self.map_shape = map_shape
        self.complete = {"robot_state":{"position":copy.copy(self.robot_initial_position), "grasping":False}, "object_state":{"position":copy.copy(self.object_initial_position)}, "map":np.zeros(map_shape)}
        self.obj_move_likelihood = oml
        self.mem_reward = None
        
    def reset(self):

        #self.complete = {"robot_state":{"position":self.robot_initial_position.copy(), "grasping":False}, "object_state":{"position":self.object_initial_position.copy()}, "map":np.zeros((6,6))}
        self.complete = {"robot_state":{"position":copy.copy(self.robot_initial_position), "grasping":False}, "object_state":{"position":copy.copy(self.object_initial_position)}, "map":np.zeros(self.map_shape)}
#        print("(Reset)", self.complete["robot_state"]["position"])
    
    def update(self, action):
        if action == "G":
            if self.complete["robot_state"]["grasping"]:
                self.complete["robot_state"]["grasping"] = False
            else:
                self.complete["robot_state"]["grasping"] = True
        else:
            try:
                if self.complete["robot_state"]["position"][0] + action[0] < 0 or self.complete["robot_state"]["position"][1] + action[1] < 0:
                    raise IndexError 
                self.complete["map"][self.complete["robot_state"]["position"][0] + action[0], self.complete["robot_state"]["position"][1] + action[1]] = 0
                self.complete["robot_state"]["position"][0] += action[0]
                self.complete["robot_state"]["position"][1] += action[1]
            except IndexError:
                #print("Won't move that way!")
                pass

        if self.complete["robot_state"]["position"] == [self.complete["object_state"]["position"][0] - 1, self.complete["object_state"]["position"][1]] and action == "G" and self.complete["robot_state"]["grasping"]:
            self.mem_reward = 100
        elif self.complete["robot_state"]["position"] == self.complete["object_state"]["position"]:
            self.mem_reward = -100
        elif self.complete["robot_state"]["position"] == [self.complete["object_state"]["position"][0] - 1, self.complete["object_state"]["position"][1]] and not action == (1,0):
            self.mem_reward = -100
        elif self.complete["robot_state"]["position"][0] == self.complete["map"].shape[0] - 1 and action == "G":
            self.mem_reward = -100
        elif self.complete["robot_state"]["position"][0] == self.complete["map"].shape[0] - 1 and action == (1,0) and not self.complete["robot_state"]["grasping"]:
            self.mem_reward = -100
        else:
            self.mem_reward = -1
        

    def reward(self):
        return self.mem_reward
        
        
    def state(self):
        if not self.mem_reward == 100:
            if random.random() < self.obj_move_likelihood:
                obj_action = random.choice([-1,1])
                if self.complete["object_state"]["position"][1] + obj_action >= 0 and self.complete["object_state"]["position"][1] + obj_action < self.complete["map"].shape[1]:
                    self.complete["object_state"]["position"][1] += obj_action
        return {"my_position":tuple(self.complete["robot_state"]["position"]), "goal_position":tuple(self.complete["object_state"]["position"])}

class Viz:
    def __init__(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.fig.patch.set_facecolor("white")  # Set overall figure background to white
        plt.show(block=False)
        time.sleep(3)
        self.bottom_note = None
        plt.rcParams['font.family'] = 'Ubuntu'
        
    def render(self, complete, info):
        fig, ax = self.fig, self.ax
        ax.clear()
        
        if self.bottom_note:
            self.bottom_note.remove()
        
        g_img = plt.imread("robot_g.png")  # image for cell value 3 (small image)
        not_g_img = plt.imread("robot_notg.png")  # slightly different image for cell value 2
        object_img = plt.imread("object.png")   # image for cell value 4 (e.g. block)
        
        zoom = 3 / max(complete["map"].shape)
        ax.imshow(complete["map"], cmap='gray', interpolation='none')
        
         # Get matrix dimensions
        num_rows, num_cols = complete["map"].shape

        # Set ticks to match each cell
        ax.set_xticks(np.arange(num_cols))
        ax.set_yticks(np.arange(num_rows))
        ax.set_xticklabels(np.arange(num_cols))
        ax.set_yticklabels(np.arange(num_rows))
        
        # Optionally, add minor ticks and grid lines to separate cells clearly
        ax.set_xticks(np.arange(-0.5, num_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, num_rows, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

        # Determine the robot's position.
        # Here we assume complete["robot_state"]["position"] is [row, col]
        robot_row = int(complete["robot_state"]["position"][0])
        robot_col = int(complete["robot_state"]["position"][1])
        
        # Choose the image based on whether the robot is grasping.
        if complete["robot_state"]["grasping"]:
            robot_img = g_img
        else:
            robot_img = not_g_img

        # Create an image box for the robot.
        # Coordinates for AnnotationBbox: (x, y) where x is the column and y is the row.
        robot_box = OffsetImage(robot_img, zoom=zoom)
        ab_robot = AnnotationBbox(robot_box, (robot_col, robot_row), frameon=False)
        ax.add_artist(ab_robot)
        
        # Determine the object's position
        object_row = int(complete["object_state"]["position"][0])
        object_col = int(complete["object_state"]["position"][1])
        
        # Create an image box for the object.
        object_box = OffsetImage(object_img, zoom=zoom)
        ab_object = AnnotationBbox(object_box, (object_col, object_row), frameon=False)
        ax.add_artist(ab_object)
        

        if info["termination"]:
            print(info["reward"])
            ax.annotate("Terminated with " + str(info["reward"]), xy=(info["termination"][1], info["termination"][0]), xytext=(np.pi/2 + 1, 0.5), arrowprops=dict(color='green', arrowstyle="->"), color='green', zorder=10)
        self.bottom_note = plt.figtext(0.5, 0.01, "Object movement likelihood: " + str(info["param"]), ha="center", fontsize=12, color='blue')
        plt.title(info["title"])

        fig.canvas.draw()
        plt.pause(0.001)  # A short pause to process GUI events

    
    def visualize(self, complete):
        depiction = complete["map"].copy()
        if complete["robot_state"]["grasping"]:
            depiction[complete["robot_state"]["position"][0], complete["robot_state"]["position"][1]] = 2
        else:
            depiction[complete["robot_state"]["position"][0], complete["robot_state"]["position"][1]] = 3
        depiction[complete["object_state"]["position"][0], complete["object_state"]["position"][1]] = 4
        print(depiction)
        print("Grasping?:", complete["robot_state"]["grasping"])
        del depiction
    
    def status(self, reward, i=-1, action=None):
        print("Reward:", reward, "\tAction was", action, "\tStep:", i+1)

class Robot:
    def __init__(self, epsilon = 0.1, with_stops=False):
        self.epsilon = epsilon
        self.actions = {"W":(-1,0), "A":(0,-1), "S":(1,0), "D":(0,1), " ":"G"}
        if with_stops:
            self.qtable = defaultdict(lambda : {(1,0):0.0, (-1,0):0.0, (0,1):0.0, (0,-1):0.0, (0,0):0.0, "G":0.0})
            self.actions["X"] = (0,0)
        else:
            self.qtable = defaultdict(lambda : {(1,0):0.0, (-1,0):0.0, (0,1):0.0, (0,-1):0.0, "G":0.0})    
    def policy(self, state, det=False):
        epsilon = self.epsilon
        if random.random() < epsilon and not det:
            return random.choice(list(self.actions.values()))
        else:
            state = state
            best_actions = []
            best_value = -math.inf
            for action in self.qtable[state].keys():
                if self.qtable[state][action] > best_value:
                    best_actions = [action]
                    best_value = self.qtable[state][action]
                elif self.qtable[state][action] == best_value:
                    best_actions.append(action)
            if len(best_actions) > 1:
                return random.choice(best_actions)
            else:
                return best_actions[0]

def learn():
    s = Sim()
    v = Viz()
    r = Robot()
    episode_length = 100
    episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
    num_episodes = 1000
    n = 3
    gamma = 0.99
    alpha = 0.1
    cum_rewards = []
    cum_steps = []
    final_rewards = []
    for l in tqdm(range(0, num_episodes)):
        t = 0
        
        state = (s.state()["my_position"], s.state()["object_position"])
        states = [state]
        actions = []
        rewards = []
        terminated = False
        termination_index = -999
        cum_reward = 0
        terminal_t = 0

        while t < episode_length + n and termination_index <= episode_length + n:
            #print("t:", t, "\tterm_index:", termination_index)
            print(str(t) + ".", state, end=", ")
            if not terminated:
                
                action = r.policy(state)
                
                actions.append(action)
                s.update(action)
                reward = s.reward()
                
                rewards.append(reward)
                cum_reward += reward
                state = s.state()["my_position"]
                states.append(state)
            else:
                action = None
            print(action, end=", ")
            print(reward)
            if t - n >= 0:
                _return = 0
                print("Pair", states[t-n], ",", actions[t-n], "gets an update at time ", t)
                for N in range(0, n):
                    if t-n+N < len(rewards):
                        print(rewards[t-n+N], gamma, N, end=" + ")        
                        _return += rewards[t-n+N]*gamma**N
                
                if terminated:
                    print("(at [", states[terminal_t], actions[terminal_t], "] =", r.qtable[states[terminal_t]][actions[terminal_t]], N+1, end=") = ")
                    _return += r.qtable[states[terminal_t]][actions[terminal_t]]*gamma**(N+1)
                else:
                    print("(at [", states[t-1], actions[t-1], "] =", r.qtable[states[t-1]][actions[t-1]], N+1, end=") = ")
                    _return += r.qtable[states[t-1]][actions[t-1]]*gamma**(N+1)
                print(_return, "\n")
                r.qtable[states[t-n]][actions[t-n]] += alpha*_return
            
            #v.visualize(s.complete)
            #v.status(reward, t, action)
            #time.sleep(0.05)
                
            if not terminated:
                if episode_end_condition(reward):
                    terminated = True
                    termination_index = episode_length
                    # Fix Q estimate at the terminal state for remaining updates
                    terminal_t = t
                    #time.sleep(1)
                    
            if terminated:
                termination_index += 1
            t += 1
        final_rewards.append(reward)   
        time.sleep(0)          
        s.reset()
        cum_steps.append(t - n)
        cum_rewards.append(cum_reward)
        
    colors = ['green' if s == 100 else 'red' if s == -100 else 'yellow' for s in final_rewards]
    print(num_episodes, len(cum_steps))
    plt.scatter(range(0, num_episodes), cum_steps, c=colors, edgecolors='black', linewidth=0.2, s=4, zorder=3)
    plt.plot(range(0, num_episodes), cum_steps)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Steps')
    plt.title('Cumulative Steps per Episode, showing Success/Failure/Timeout')
    plt.show()
    
    fig2 = plt.figure()
    plt.plot(range(0, num_episodes), cum_rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward per Episode')
    plt.show()
    return r        

################################################################################################################
#
#          !!!!!!!!!!!!!!!!!!!!!1 EXPERIMENTAL GOAL LEARNING

def test_learn_strict():
    sample_size = 100 # Make 1 if just capturing visual.
    avg_cum_steps = []
    tot_final_rewards = []
    final_rewards = {0.0: [], 0.1: [], 0.5:[], 0.8:[], 1.0:[]}
        #final_rewards = {0.0: []}
    v = Viz()
    for b in tqdm(range(0, sample_size)):
    
        #v = Viz()
        r = Robot() # keep the robot outside the configuring
        episode_length = 200
        episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
        num_episodes = 2000
        n = 5
        gamma = 0.9
        alpha = 0.01 # old is 0.01

        
        
        for object_init in [[5,5], [5,4], [5,3], [5,2], [5,1], [5,0]]:
            for r_init in [[0,0], [5,0], [0, 5], [5,5]]:
                s = Sim(0.0, r_init, object_init)
                for l in range(0, num_episodes):
                    t = 0
                    state = (s.state()["my_position"], s.state()["goal_position"])
                    states = [state]
                    actions = []
                    rewards = []
                    terminated = False
                    termination_index = -999
                    cum_reward = 0
                    terminal_t = 0

                    while t < episode_length + n and termination_index <= episode_length + n:
                        #print("t:", t, "\tterm_index:", termination_index)
                        #print(str(t) + ".", state, end=", ")
                        if not terminated:
                            
                            action = r.policy(state)
                            
                            actions.append(action)
                            s.update(action)
                            reward = s.reward()
                            
                            rewards.append(reward)
                            cum_reward += reward
                            state = (s.state()["my_position"], s.state()["goal_position"])
                            states.append(state)
                        else:
                            action = None
                        #print(action, end=", ")
                        #print(reward)
                        if t - n >= 0:
                            _return = 0
                            #print("Pair", states[t-n], ",", actions[t-n], "gets an update at time ", t)
                            N=0
                            for N in range(0, n):
                                if t-n+N < len(rewards):
                                    #print(rewards[t-n+N], gamma, N, end=" + ")        
                                    _return += rewards[t-n+N]*gamma**N
                            
                            if terminated:
                                #print("(at [", states[terminal_t], actions[terminal_t], "] =", r.qtable[states[terminal_t]][actions[terminal_t]], N+1, end=") = ")
                                _return += r.qtable[states[terminal_t]][actions[terminal_t]]*gamma**(N+1)
                            else:
                                #print("(at [", states[t-1], actions[t-1], "] =", r.qtable[states[t-1]][actions[t-1]], N+1, end=") = ")
                                _return += r.qtable[states[t-1]][actions[t-1]]*gamma**(N+1)
                            #print(_return, "\n")
                            r.qtable[states[t-n]][actions[t-n]] += alpha*_return
                        
                        #v.visualize(s.complete)
                        #v.status(reward, t, action)
                        #time.sleep(0.05)
                            
                        if not terminated:
                            if episode_end_condition(reward):
                                terminated = True
                                termination_index = episode_length
                                # Fix Q estimate at the terminal state for remaining updates
                                terminal_t = t
                                #time.sleep(1)
                                
                        if terminated:
                            termination_index += 1
                        t += 1
                    s.reset()
                    # end episode
        
        e= 100

        episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
        for param in final_rewards.keys():
            terminated = False
            #print("***************8", param, "*****************")
            info = {"param":param, "termination":None, "title":"After " + str(num_episodes) + " episodes of training:"}
            s = Sim(param, [0,3], [5,2])
            
            state = (s.state()["my_position"], s.state()["goal_position"])
            #v.render(s.complete, info)
            #time.sleep(2)
            while t < e and not terminated:
                #time.sleep(0.2)
                
                action = r.policy(state)
                s.update(action)
                reward = s.reward()
                state = (s.state()["my_position"], s.state()["goal_position"])
                if episode_end_condition(reward):
                    #print(reward)
                    info["termination"] = state[1]
                    info["reward"] = reward
                    #v.render(s.complete, info)
                    #time.sleep(2)
                    terminated = True
                    
                #v.render(s.complete, info)
            s.reset()
            final_rewards[param].append(reward)
    #print(final_rewards)        
    plt.ioff()
    fig, ax = plt.subplots()
    x = np.arange(len(final_rewards.keys()))
    bottom = np.zeros(len(final_rewards.keys()))
    for i, param_key in enumerate(final_rewards.keys()):
        print(param_key, "has:")
        final_neg_rewards = final_rewards[param_key].count(-100)
        ax.bar(x[i], final_neg_rewards, bottom=0, color='red')
        bottom += final_neg_rewards
        print(final_neg_rewards)
        final_neut_rewards = final_rewards[param_key].count(-1)
        ax.bar(x[i], final_neut_rewards, bottom=final_neg_rewards, color='yellow')
        bottom += final_neut_rewards
        print(final_neut_rewards)
        final_pos_rewards = final_rewards[param_key].count(100)
        ax.bar(x[i], final_pos_rewards, bottom=final_neg_rewards + final_neut_rewards, color='green')
        print(final_pos_rewards)
        
    # end k samples
    
    plt.xlabel('Object movement likelihood')
    plt.ylabel('Final reward')
    plt.title('Average final reward over ' + str(sample_size) + ' samples')
    ax.set_xticks(x)
    ax.set_xticklabels([label for label in final_rewards.keys()])
    ax.legend()
    plt.show()

def sample_learn_strict():
    sample_size = 10
    avg_cum_steps = []
    tot_final_rewards = []
    for b in tqdm(range(0, sample_size)):
    
        #v = Viz()
        r = Robot() # keep the robot outside the configuring
        episode_length = 200
        episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
        num_episodes = 2000
        n = 5
        gamma = 0.9
        alpha = 0.01 # old is 0.01
        cum_rewards = []
        cum_steps = []
        final_rewards = []
        for object_init in [[5,5], [5,4], [5,3], [5,2], [5,1], [5,0]]:
            for r_init in [[0,0], [5,0], [0, 5], [5,5]]:
                s = Sim(0.0, r_init, object_init)
                for l in range(0, num_episodes):
                    t = 0
                    state = (s.state()["my_position"], s.state()["goal_position"])
                    states = [state]
                    actions = []
                    rewards = []
                    terminated = False
                    termination_index = -999
                    cum_reward = 0
                    terminal_t = 0

                    while t < episode_length + n and termination_index <= episode_length + n:
                        #print("t:", t, "\tterm_index:", termination_index)
                        #print(str(t) + ".", state, end=", ")
                        if not terminated:
                            
                            action = r.policy(state)
                            
                            actions.append(action)
                            s.update(action)
                            reward = s.reward()
                            
                            rewards.append(reward)
                            cum_reward += reward
                            state = (s.state()["my_position"], s.state()["goal_position"])
                            states.append(state)
                        else:
                            action = None
                        #print(action, end=", ")
                        #print(reward)
                        if t - n >= 0:
                            _return = 0
                            #print("Pair", states[t-n], ",", actions[t-n], "gets an update at time ", t)
                            N=0
                            for N in range(0, n):
                                if t-n+N < len(rewards):
                                    #print(rewards[t-n+N], gamma, N, end=" + ")        
                                    _return += rewards[t-n+N]*gamma**N
                            
                            if terminated:
                                #print("(at [", states[terminal_t], actions[terminal_t], "] =", r.qtable[states[terminal_t]][actions[terminal_t]], N+1, end=") = ")
                                _return += r.qtable[states[terminal_t]][actions[terminal_t]]*gamma**(N+1)
                            else:
                                #print("(at [", states[t-1], actions[t-1], "] =", r.qtable[states[t-1]][actions[t-1]], N+1, end=") = ")
                                _return += r.qtable[states[t-1]][actions[t-1]]*gamma**(N+1)
                            #print(_return, "\n")
                            r.qtable[states[t-n]][actions[t-n]] += alpha*_return
                        
                        #v.visualize(s.complete)
                        #v.status(reward, t, action)
                        #time.sleep(0.05)
                            
                        if not terminated:
                            if episode_end_condition(reward):
                                terminated = True
                                termination_index = episode_length
                                # Fix Q estimate at the terminal state for remaining updates
                                terminal_t = t
                                #time.sleep(1)
                                
                        if terminated:
                            termination_index += 1
                        t += 1
                    final_rewards.append(reward)   
                    time.sleep(0)          
                    s.reset()
                    cum_steps.append(t - n)
                    cum_rewards.append(cum_reward)
                    # end episode
        avg_cum_steps.append(cum_steps)
        tot_final_rewards.append(final_rewards)
    _avg_cum_steps = np.mean(avg_cum_steps, axis=0)

    mode_final_rewards = mode(tot_final_rewards, axis=0)[0]

    colors = ['green' if s == 100 else 'red' if s == -100 else 'yellow' for s in mode_final_rewards]
    plt.scatter(range(0, num_episodes*24), _avg_cum_steps, c=colors, edgecolors='black', linewidth=0.2, s=4, zorder=3)
    plt.plot(range(0, num_episodes*24), _avg_cum_steps)
    # end k samples
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Steps')
    plt.title('Average Cumulative Steps per Episode over ' + str(sample_size) + ' Runs, showing mode Success/Failure/Timeout')
    #plt.legend()
    plt.show()
    # end episodes for learner
def goal_learn_strict():
    
    #v = Viz()
    r = Robot() # keep the robot outside the configuring
    episode_length = 200
    episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
    num_episodes = 2000
    n = 5
    gamma = 0.9
    alpha = 0.01 # old is 0.01
    cum_rewards = []
    cum_steps = []
    final_rewards = []
    for object_init in [[5,5], [5,4], [5,3], [5,2], [5,1], [5,0]]:
        for r_init in [[0,0], [5,0], [0, 5], [5,5]]:
            s = Sim(0.0, r_init, object_init)
            for l in range(0, num_episodes):
                t = 0
                state = (s.state()["my_position"], s.state()["goal_position"])
                states = [state]
                actions = []
                rewards = []
                terminated = False
                termination_index = -999
                cum_reward = 0
                terminal_t = 0

                while t < episode_length + n and termination_index <= episode_length + n:
                    #print("t:", t, "\tterm_index:", termination_index)
                    #print(str(t) + ".", state, end=", ")
                    if not terminated:
                        
                        action = r.policy(state)
                        
                        actions.append(action)
                        s.update(action)
                        reward = s.reward()
                        
                        rewards.append(reward)
                        cum_reward += reward
                        state = (s.state()["my_position"], s.state()["goal_position"])
                        states.append(state)
                    else:
                        action = None
                    #print(action, end=", ")
                    #print(reward)
                    if t - n >= 0:
                        _return = 0
                        print("Pair", states[t-n], ",", actions[t-n], "gets an update at time ", t)
                        N=0
                        for N in range(0, n):
                            if t-n+N < len(rewards):
                                #print(rewards[t-n+N], gamma, N, end=" + ")        
                                _return += rewards[t-n+N]*gamma**N
                        
                        if terminated:
                            #print("(at [", states[terminal_t], actions[terminal_t], "] =", r.qtable[states[terminal_t]][actions[terminal_t]], N+1, end=") = ")
                            _return += r.qtable[states[terminal_t]][actions[terminal_t]]*gamma**(N+1)
                        else:
                            #print("(at [", states[t-1], actions[t-1], "] =", r.qtable[states[t-1]][actions[t-1]], N+1, end=") = ")
                            _return += r.qtable[states[t-1]][actions[t-1]]*gamma**(N+1)
                        #print(_return, "\n")
                        r.qtable[states[t-n]][actions[t-n]] += alpha*_return
                    
                    #v.visualize(s.complete)
                    #v.status(reward, t, action)
                    #time.sleep(0.05)
                        
                    if not terminated:
                        if episode_end_condition(reward):
                            terminated = True
                            termination_index = episode_length
                            # Fix Q estimate at the terminal state for remaining updates
                            terminal_t = t
                            #time.sleep(1)
                            
                    if terminated:
                        termination_index += 1
                    t += 1
                final_rewards.append(reward)   
                time.sleep(0)          
                s.reset()
                cum_steps.append(t - n)
                cum_rewards.append(cum_reward)
                # end episode
    colors = ['green' if s == 100 else 'red' if s == -100 else 'yellow' for s in final_rewards]
    print(num_episodes, len(cum_steps))
    plt.scatter(range(0, num_episodes*24), cum_steps, c=colors, edgecolors='black', linewidth=0.2, s=4, zorder=3)
    plt.plot(range(0, num_episodes*24), cum_steps)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Steps')
    plt.title('Cumulative Steps per Episode, showing Success/Failure/Timeout')
    plt.show()
    return r
    # end episodes for learner

def goal_learn():
    s = Sim(1.0)
    #v = Viz()
    r = Robot()
    episode_length = 200
    episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
    num_episodes = 2000
    n = 5
    gamma = 0.9
    alpha = 0.01 # old is 0.01
    cum_rewards = []
    cum_steps = []
    final_rewards = []
    for l in range(0, num_episodes):
        t = 0
        state = (s.state()["my_position"], s.state()["goal_position"])
        states = [state]
        actions = []
        rewards = []
        terminated = False
        termination_index = -999
        cum_reward = 0
        terminal_t = 0

        while t < episode_length + n and termination_index <= episode_length + n:
            #print("t:", t, "\tterm_index:", termination_index)
            #print(str(t) + ".", state, end=", ")
            if not terminated:
                
                action = r.policy(state)
                
                actions.append(action)
                s.update(action)
                reward = s.reward()
                
                rewards.append(reward)
                cum_reward += reward
                state = (s.state()["my_position"], s.state()["goal_position"])
                states.append(state)
            else:
                action = None
            #print(action, end=", ")
            #print(reward)
            if t - n >= 0:
                _return = 0
                #print("Pair", states[t-n], ",", actions[t-n], "gets an update at time ", t)
                N=0
                for N in range(0, n):
                    if t-n+N < len(rewards):
                        #print(rewards[t-n+N], gamma, N, end=" + ")        
                        _return += rewards[t-n+N]*gamma**N
                
                if terminated:
                    #print("(at [", states[terminal_t], actions[terminal_t], "] =", r.qtable[states[terminal_t]][actions[terminal_t]], N+1, end=") = ")
                    _return += r.qtable[states[terminal_t]][actions[terminal_t]]*gamma**(N+1)
                else:
                    #print("(at [", states[t-1], actions[t-1], "] =", r.qtable[states[t-1]][actions[t-1]], N+1, end=") = ")
                    _return += r.qtable[states[t-1]][actions[t-1]]*gamma**(N+1)
                #print(_return, "\n")
                r.qtable[states[t-n]][actions[t-n]] += alpha*_return
            
            #v.visualize(s.complete)
            #v.status(reward, t, action)
            #time.sleep(0.05)
                
            if not terminated:
                if episode_end_condition(reward):
                    terminated = True
                    termination_index = episode_length
                    # Fix Q estimate at the terminal state for remaining updates
                    terminal_t = t
                    #time.sleep(1)
                    
            if terminated:
                termination_index += 1
            t += 1
        final_rewards.append(reward)   
        time.sleep(0)          
        s.reset()
        cum_steps.append(t - n)
        cum_rewards.append(cum_reward)
        # end episode
    colors = ['green' if s == 100 else 'red' if s == -100 else 'yellow' for s in final_rewards]
    print(num_episodes, len(cum_steps))
    plt.scatter(range(0, num_episodes), cum_steps, c=colors, edgecolors='black', linewidth=0.2, s=4, zorder=3)
    plt.plot(range(0, num_episodes), cum_steps)
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Steps')
    plt.title('Cumulative Steps per Episode, showing Success/Failure/Timeout')
    plt.show()
    return r
    # end episodes for learner
    
def sample_goal_learn():
    
    #   [n, num_episodes, alpha]
    params = [True, False]
    for param in params: # change for tuning
        sample_size = 200
        avg_cum_steps = []
        tot_final_rewards = []
        for b in tqdm(range(0, sample_size)):
            s = Sim(0.5)
            #v = Viz()
            r = Robot(with_stops=param)
            episode_length = 200
            episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
            num_episodes = 2000
            n = 5
            gamma = 0.9
            alpha = 0.01 # old is 0.01
            cum_rewards = []
            cum_steps = []
            final_rewards = []
            for l in range(0, num_episodes):
                t = 0
                state = (s.state()["my_position"], s.state()["goal_position"])
                states = [state]
                actions = []
                rewards = []
                terminated = False
                termination_index = -999
                cum_reward = 0
                terminal_t = 0

                while t < episode_length + n and termination_index <= episode_length + n:
                    #print("t:", t, "\tterm_index:", termination_index)
                    #print(str(t) + ".", state, end=", ")
                    if not terminated:
                        
                        action = r.policy(state)
                        
                        actions.append(action)
                        s.update(action)
                        reward = s.reward()
                        
                        rewards.append(reward)
                        cum_reward += reward
                        state = (s.state()["my_position"], s.state()["goal_position"])
                        states.append(state)
                    else:
                        action = None
                    #print(action, end=", ")
                    #print(reward)
                    if t - n >= 0:
                        _return = 0
                        #print("Pair", states[t-n], ",", actions[t-n], "gets an update at time ", t)
                        N=0
                        for N in range(0, n):
                            if t-n+N < len(rewards):
                                #print(rewards[t-n+N], gamma, N, end=" + ")        
                                _return += rewards[t-n+N]*gamma**N
                        
                        if terminated:
                            #print("(at [", states[terminal_t], actions[terminal_t], "] =", r.qtable[states[terminal_t]][actions[terminal_t]], N+1, end=") = ")
                            _return += r.qtable[states[terminal_t]][actions[terminal_t]]*gamma**(N+1)
                        else:
                            #print("(at [", states[t-1], actions[t-1], "] =", r.qtable[states[t-1]][actions[t-1]], N+1, end=") = ")
                            _return += r.qtable[states[t-1]][actions[t-1]]*gamma**(N+1)
                        #print(_return, "\n")
                        r.qtable[states[t-n]][actions[t-n]] += alpha*_return
                    
                    #v.visualize(s.complete)
                    #v.status(reward, t, action)
                    #time.sleep(0.05)
                        
                    if not terminated:
                        if episode_end_condition(reward):
                            terminated = True
                            termination_index = episode_length
                            # Fix Q estimate at the terminal state for remaining updates
                            terminal_t = t
                            #time.sleep(1)
                            
                    if terminated:
                        termination_index += 1
                    t += 1
                final_rewards.append(reward)   
                time.sleep(0)          
                s.reset()
                cum_steps.append(t - n)
                cum_rewards.append(cum_reward)
                # end episode
            avg_cum_steps.append(cum_steps)
            tot_final_rewards.append(final_rewards)
            # end episodes for learner
        _avg_cum_steps = np.mean(avg_cum_steps, axis=0)
        print(_avg_cum_steps)
        mode_final_rewards = mode(tot_final_rewards, axis=0)[0]
        print(mode_final_rewards)
        colors = ['green' if s == 100 else 'red' if s == -100 else 'yellow' for s in mode_final_rewards]
        plt.scatter(range(0, num_episodes), _avg_cum_steps, c=colors, edgecolors='black', linewidth=0.2, s=4, zorder=3)
        plt.plot(range(0, num_episodes), _avg_cum_steps, label='With stops = '+str(param))
        # end k samples
    # end sample configurations
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Steps')
    plt.title('Average Cumulative Steps per Episode over ' + str(sample_size) + ' Runs, showing mode Success/Failure/Timeout')
    plt.legend()
    plt.show()
    
    return r  

def sample_learn():
    
    
    
    for param in [0.0]: # change for tuning
        sample_size = 500
        avg_cum_steps = []
        tot_final_rewards = []
        for b in tqdm(range(0, sample_size)):
            s = Sim()
            #v = Viz()
            r = Robot()
            episode_length = 200
            episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
            num_episodes = 500
            n = 5
            gamma = 0.9
            alpha = 0.02
            cum_rewards = []
            cum_steps = []
            final_rewards = []
            for l in range(0, num_episodes):
                t = 0
                state = s.state()["my_position"]
                states = [state]
                actions = []
                rewards = []
                terminated = False
                termination_index = -999
                cum_reward = 0
                terminal_t = 0

                while t < episode_length + n and termination_index <= episode_length + n:
                    #print("t:", t, "\tterm_index:", termination_index)
                    #print(str(t) + ".", state, end=", ")
                    if not terminated:
                        
                        action = r.policy(state)
                        
                        actions.append(action)
                        s.update(action)
                        reward = s.reward()
                        
                        rewards.append(reward)
                        cum_reward += reward
                        state = s.state()["my_position"]
                        states.append(state)
                    else:
                        action = None
                    #print(action, end=", ")
                    #print(reward)
                    if t - n >= 0:
                        _return = 0
                        #print("Pair", states[t-n], ",", actions[t-n], "gets an update at time ", t)
                        N=0
                        for N in range(0, n):
                            if t-n+N < len(rewards):
                                #print(rewards[t-n+N], gamma, N, end=" + ")        
                                _return += rewards[t-n+N]*gamma**N
                        
                        if terminated:
                            #print("(at [", states[terminal_t], actions[terminal_t], "] =", r.qtable[states[terminal_t]][actions[terminal_t]], N+1, end=") = ")
                            _return += r.qtable[states[terminal_t]][actions[terminal_t]]*gamma**(N+1)
                        else:
                            #print("(at [", states[t-1], actions[t-1], "] =", r.qtable[states[t-1]][actions[t-1]], N+1, end=") = ")
                            _return += r.qtable[states[t-1]][actions[t-1]]*gamma**(N+1)
                        #print(_return, "\n")
                        r.qtable[states[t-n]][actions[t-n]] += alpha*_return
                    
                    #v.visualize(s.complete)
                    #v.status(reward, t, action)
                    #time.sleep(0.05)
                        
                    if not terminated:
                        if episode_end_condition(reward):
                            terminated = True
                            termination_index = episode_length
                            # Fix Q estimate at the terminal state for remaining updates
                            terminal_t = t
                            #time.sleep(1)
                            
                    if terminated:
                        termination_index += 1
                    t += 1
                final_rewards.append(reward)   
                time.sleep(0)          
                s.reset()
                cum_steps.append(t - n)
                cum_rewards.append(cum_reward)
                # end episode
            avg_cum_steps.append(cum_steps)
            tot_final_rewards.append(final_rewards)
            # end episodes for learner
        _avg_cum_steps = np.mean(avg_cum_steps, axis=0)
        #print(_avg_cum_steps)
        mode_final_rewards = mode(tot_final_rewards, axis=0)[0]
        #print(mode_final_rewards)
        colors = ['green' if s == 100 else 'red' if s == -100 else 'yellow' for s in mode_final_rewards]
        plt.scatter(range(0, num_episodes), _avg_cum_steps, c=colors, edgecolors='black', linewidth=0.2, s=4, zorder=3)
        #plt.plot(range(0, num_episodes), _avg_cum_steps, label='gamma='+str(param)) # Use if tuning parameters
        plt.plot(range(0, num_episodes), _avg_cum_steps)
        # end k samples
    # end sample configurations
    
    plt.xlabel('Episodes')
    plt.ylabel('Average Cumulative Steps')
    plt.title('Average Cumulative Steps per Episode over ' + str(sample_size) + ' Runs, showing mode Success/Failure/Timeout')
    #plt.legend() #use if fine tuning
    plt.show()
    
    return r        


#############################################################################################3
def teleop():
    s = Sim()
    v = Viz()
    r = Robot()
    # state = s.state()
    print("WASD move <space> to grab.")
    while True:
        # action = policy(state)
        action = r.actions[input(">>> ").upper()]
        s.update(action)
        reward = s.reward()
        # state = s.state()
        
        v.render(s.complete)
        v.status(reward, 0, action)
        
def watch_random_policy():
    
    s = Sim()
    v = Viz()
    r = Robot()
    episode_length = 10
    episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
    try:
        while True:
            state = s.state()["my_position"]
            for i in range(0, episode_length):
                action = r.policy(state)
                s.update(action)
                reward = s.reward()
                state = s.state()["my_position"]
                
                v.visualize(s.complete)
                v.status(reward, i, action)
                time.sleep(0.25)
                if episode_end_condition(reward):
                    break
                   
            time.sleep(1)
            s.reset()

    except KeyboardInterrupt:
        print("Exiting.")

def test_dynamics(r):
    params = [0.0, 0.1, 0.5, 0.8, 1.0]
    fig, ax = plt.subplots()
    x = np.arange(len(params))
    
    episode_length = 100
    terminated = False
    episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
    final_rewards = []
    for param in params:
        s = Sim(param, [0,3], [5,2])
        state = (s.state()["my_position"], s.state()["goal_position"])
        while t < episode_length and not terminated:
            action = r.policy(state)
            s.update(action)
            reward = s.reward()
            state = (s.state()["my_position"], s.state()["goal_position"])
            if episode_end_condition(reward):
                terminated = True
        s.reset()

             
def watch_robot(r):
    
    s = Sim(1.0)
    v = Viz()
    episode_length = 1000
    episode_end_condition = lambda r: r in [-100, 100] # using reward as signal for episode end conditions
    try:
        while True:
            state = s.state()["my_position"]
            for i in range(0, episode_length):
                action = r.policy(state, det=True)
                print("pi(",state,") =", action) 
                s.update(action)
                reward = s.reward()
                state = s.state()["my_position"]
                
                v.render(s.complete)
                v.status(reward, i, action)
                time.sleep(0.25)
                if episode_end_condition(reward):
                    break
                   
            time.sleep(1)
            s.reset()

    except KeyboardInterrupt:
        print("Exiting.")
        
        
if __name__ == "__main__":
    #watch_random_policy()
    #teleop()
    #r = sample_learn_strict()
    r = test_learn_strict()

    
