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

    
