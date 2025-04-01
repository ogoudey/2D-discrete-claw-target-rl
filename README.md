# Reinforcement Learning on a Claw in a Simple 2D Discrete Environment

[Video](https://www.youtube.com/watch?v=aPuqniXXhhk)

### Instructions for running
After cloning this repository, install dependencies.

Then, all operations run with the same command:
```
python3 hw3.py
```
In the code (`hw3.py`), changing the contents of `if __name__ == "__main__": ...` will change what happens.

The functions are:
- learn() -- runs the TD(n) algorithm, returning a robot with a policy.
- watch_robot(r) -- takes a robot and visualizes it.
- watch_random_policy() -- just to see what the environment is like.
- teleop() -- lets the user control the claw. The episode will not end, but you can experience the reward structure. Teleoperation is equivalent to the learning agent's environment w.r.t. the actions, etc.
- sample_learn() -- generates plots by sampling learn()-like runs.
- goal_learn() -- attempt 1 at learning a goal (didn't work too well).
- sample_goal_learn() -- generates plots for attempt 1.
- goal_learn_strict() -- attempt 2 at learning a goal.
- sample_learn_strict() -- for generating plots for attempt 2
- test_learn_strict() -- generates bar graphs. Also provided material for video. Comment out any `v.render` and `time.sleep()` lines for generating graphs.

To run the material behind the video, run test_learn_strict().
#### Code Structure
There are three supporting classes: **Robot**, **Sim**, and **Viz**. **Robot** is responsible for providing the actions and holding a policy. **Sim** provides the map, gets updated, and provides state information to the robot of its position and the target object's position. **Viz** is a class that provides various visualization features, from rendering with images to command-line visualization.
