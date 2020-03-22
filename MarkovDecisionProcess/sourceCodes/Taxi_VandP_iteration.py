import gym

import matplotlib.pyplot as plt
import numpy as np
import os
import time

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        #print ('%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0))
        print ('function took %0.3f ms', (time2-time1)*1000.0)
        return ret
    return wrap

def evaluate_rewards_and_transitions(problem, mutate=False):
    # Enumerate state and action space sizes
    num_states = problem.observation_space.n
    num_actions = problem.action_space.n

    # Intiailize T and R matrices
    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    # Iterate over states, actions, and transitions
    for state in range(num_states):
        for action in range(num_actions):
            for transition in problem.env.P[state][action]:
                probability, next_state, reward, done = transition
                R[state, action, next_state] = reward
                T[state, action, next_state] = probability
                #print("Reward: ", reward)

            # Normalize T across state + action axes
            T[state, action, :] /= np.sum(T[state, action, :])

    # Conditionally mutate and return
    if mutate:
        problem.env.R = R
        problem.env.T = T
    #print("Reward: ", np.shape(R))
    return R, T

@timing
def value_iteration(problem, gamma, R=None, T=None, max_iterations=10**6, delta=10**-3):
    """ Runs Value Iteration on a gym problem """
    value_fn = np.zeros(problem.observation_space.n)
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions(problem)

    for i in range(max_iterations):
        previous_value_fn = value_fn.copy()
        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        value_fn = np.max(Q, axis=1)

        if np.max(np.abs(value_fn - previous_value_fn)) < delta:
            break

    # Get and return optimal policy
    policy = np.argmax(Q, axis=1)
    return policy, i + 1

def encode_policy(policy, shape):
    """ One-hot encodes a policy """
    encoded_policy = np.zeros(shape)
    encoded_policy[np.arange(shape[0]), policy] = 1
    return encoded_policy

@timing
def policy_iteration(problem,  gamma, R=None, T=None, max_iterations=10**6, delta=10**-3):
    """ Runs Policy Iteration on a gym problem """
    num_spaces = problem.observation_space.n
    num_actions = problem.action_space.n

    # Initialize with a random policy and initial value function
    policy = np.array([problem.action_space.sample() for _ in range(num_spaces)])
    value_fn = np.zeros(num_spaces)

    # Get transitions and rewards
    if R is None or T is None:
        R, T = evaluate_rewards_and_transitions(problem)

    #print("reward: ", R)
    # Iterate and improve policies
    for i in range(max_iterations):
        previous_policy = policy.copy()

        for j in range(max_iterations):
            previous_value_fn = value_fn.copy()
            Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
            value_fn = np.sum(encode_policy(policy, (num_spaces, num_actions)) * Q, 1)

            if np.max(np.abs(previous_value_fn - value_fn)) < delta:
                break

        Q = np.einsum('ijk,ijk -> ij', T, R + gamma * value_fn)
        policy = np.argmax(Q, axis=1)

        if np.array_equal(policy, previous_policy):
            break

    # Return optimal policy
    return policy, i + 1

def print_policy(policy, mapping=None, shape=(0,)):
    print (np.array([mapping[action] for action in policy]).reshape(shape))
    

def run_episode(env, policy, gamma, render = False):
    """ Evaluates policy by using it to run an episode and finding its
    total reward.

    args:
    env: gym environment.
    policy: the policy to be used.
    gamma: discount factor.
    render: boolean to turn rendering on/off.

    returns:
    total reward: real value of the total reward recieved by agent under policy.
    """
    obs = env.reset()
    total_reward = 0
    step_idx = 0
    while True:
        if render:
            env.render()
        obs, reward, done , _ = env.step(int(policy[obs]))
        total_reward += (gamma ** step_idx * reward)
        step_idx += 1
        if done:
            break
    return total_reward

def evaluate_policy(env, policy, gamma,  n = 100):
    """ Evaluates a policy by running it n times.
    returns:
    average total reward
    """
    scores = [
            run_episode(env, policy, gamma = gamma, render = False)
            for _ in range(n)]
    return scores

def run_discrete(environment_name, mapping, gamma, shape=None):
    problem = gym.make(environment_name)
    print ('== {} =='.format(environment_name))
    print ('Actions:', problem.env.action_space.n)
    print ('States:', problem.env.observation_space.n)
    print (problem.env.desc)
    print

    print ('== Value Iteration == (\u03B3=%0.2f)' %gamma)
    value_policy, iters = value_iteration(problem, gamma)
    scores = evaluate_policy(problem, value_policy, gamma, n=100)
    print ('Iterations:', iters)
    print('Policy average score = ', np.mean(scores))
    print
    episodes= np.array(range(1,len(scores)+1))
    plot_figures(scores, episodes, gamma,'Value Iteration')

    print ('== Policy Iteration ==(\u03B3=%0.2f)' %gamma)
    policy, iters = policy_iteration(problem, gamma)
    scores = evaluate_policy(problem, policy, gamma, n=100)
    print ('Iterations:', iters)
    print('Policy average score = ', np.mean(scores))
    print
    
    episodes= np.array(range(1,len(scores)+1))
    plot_figures(scores, episodes, gamma, 'Policy Iteration')

    diff = sum([abs(x-y) for x, y in zip(policy.flatten(), value_policy.flatten())])
    if diff > 0:
        print ('Discrepancy:', diff)
        print

    if shape is not None:
        print ('== Policy ==')
        print_policy(policy, mapping, shape)
        print

    return policy

def plot_figures(scores, episodes, gamma, iterType):
    dir = os.path.join("./","A4_Plots")
    if not os.path.exists(dir):
        os.mkdir(dir)
    plt.figure()
    plt.plot(episodes, scores, 'go', label='Reward')
    plt.title('Reward vs Episode ($%s: \gamma=%0.2f$)' %(iterType, gamma), fontsize=16);
    plt.ylabel('Reward', fontsize=14);
    plt.xlabel('Episode', fontsize=14);
    plt.legend(loc='best', fontsize=12)
    plt.savefig('./A4_Plots/%sTaxi_Reward_g%0.2f.png' %(iterType, gamma), bbox_inches='tight')
    plt.show()
    


#gamma=0.95
# FROZEN LAKE SMALL
#mapping = {0: "L", 1: "D", 2: "R", 3: "U"}
#shape = (4, 4)
#run_discrete('FrozenLake-v0', mapping, shape)

# FROZEN LAKE LARGE
#shape = (8, 8)
#run_discrete('FrozenLake8x8-v0', mapping ,shape)

# TAXI
mapping = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
run_discrete('Taxi-v3', mapping, gamma=0.90)
run_discrete('Taxi-v3', mapping, gamma=0.95)
run_discrete('Taxi-v3', mapping, gamma=0.50)
