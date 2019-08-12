### MDP Value Iteration and Policy Iteration

import numpy as np
import gym
import time
from lake_envs import *
#from frozen_lake import *
from frozen_lake import FrozenLakeEnv

# This makes Windows 10 console recognize escape sequences (colored output in env.render())
# https://stackoverflow.com/questions/16755142/how-to-make-win32-console-recognize-ansi-vt100-escape-sequences
import os
os.system('')

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

    P: nested dictionary
        From gym.core.Environment
        For each pair of states in [0, nS-1] and actions in [0, nA-1], P[state][action] is a
        tuple of the form (probability, nextstate, reward, terminal) where
            - probability: float
                the probability of transitioning from "state" to "nextstate" with "action"
            - nextstate: int
                denotes the state we transition to (in range [0, nS - 1])
            - reward: int
                either 0 or 1, the reward for transitioning from "state" to
                "nextstate" with "action"
            - terminal: bool
              True when "nextstate" is a terminal state (hole or goal), False otherwise
    nS: int
        number of states in the environment
    nA: int
        number of actions in the environment
    gamma: float
        Discount factor. Number in range [0, 1)
"""
policy_eval_iter_count = []
def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
    """Evaluate the value function from a given policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    policy: np.array[nS]
        The policy to evaluate. Maps states to actions.
    tol: float
        Terminate policy evaluation when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns
    -------
    value_function: np.ndarray[nS]
        The value function of the given policy, where value_function[s] is
        the value of state s
    """    
    value_function = np.zeros(nS)

    ############################
    # YOUR IMPLEMENTATION HERE #
    global policy_eval_iter_count
    v_new = np.zeros(nS)
    ok = False
    n_iter = 0
    while not ok:
        n_iter += 1
        for s in range(nS):
            a = policy[s]
            target = 0
            for prob, s2, rw, terminal in P[s][a]:
                target += prob*(rw + gamma*value_function[s2])
            v_new[s] = target
        if np.abs(v_new - value_function).max() < tol:
            ok = True
        value_function = v_new.copy()
    policy_eval_iter_count.append(n_iter)
    ############################

    return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
    """Given the value function from policy improve the policy.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    value_from_policy: np.ndarray
        The value calculated from the policy
    policy: np.array
        The previous policy.

    Returns
    -------
    new_policy: np.ndarray[nS]
        An array of integers. Each integer is the optimal action to take
        in that state according to the environment dynamics and the
        given value function.
    """

    new_policy = np.zeros(nS, dtype='int')

    ############################
    # YOUR IMPLEMENTATION HERE #
    for s in range(nS):
        best_a = None
        best_q = None
        for a in range(nA):
            q = 0.0
            for prob, s2, rw, terminal in P[s][a]:
                q += prob*(rw + gamma*value_from_policy[s2])
            if (best_q is None) or (q > best_q):
                best_q = q
                best_a = a

        new_policy[s] = best_a
    ############################
    return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
    """Runs policy iteration.

    You should call the policy_evaluation() and policy_improvement() methods to
    implement this method.

    Parameters
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        tol parameter used in policy_evaluation()
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    
    ############################
    # YOUR IMPLEMENTATION HERE #
    ok = False
    while not ok:
        value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
        pi_new = policy_improvement(P, nS, nA, value_function, policy)
        if (policy == pi_new).all():
            ok = True
        policy = pi_new.copy()
    value_function = policy_evaluation(P, nS, nA, policy, gamma, tol)
    ############################
    
    return value_function, policy

n_value_iter = 0
def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
    """
    Learn value function and policy by using value iteration method for a given
    gamma and environment.

    Parameters:
    ----------
    P, nS, nA, gamma:
        defined at beginning of file
    tol: float
        Terminate value iteration when
            max |value_function(s) - prev_value_function(s)| < tol
    Returns:
    ----------
    value_function: np.ndarray[nS]
    policy: np.ndarray[nS]
    """

    value_function = np.zeros(nS)
    policy = np.zeros(nS, dtype=int)
    ############################
    # YOUR IMPLEMENTATION HERE #
    global n_value_iter
    n_value_iter = 0
    v_new = np.zeros(nS)
    ok = False
    while not ok:
        n_value_iter += 1
        for s in range(nS):
            best_a = None
            best_q = None
            for a in range(nA):
                q = 0
                for prob, s2, rw, terminal in P[s][a]:
                    q += prob*(rw + gamma*value_function[s2])
                if (best_q is None) or q > best_q:
                    best_q = q
                    best_a = a

            v_new[s] = best_q
            policy[s] = best_a

        if np.abs(v_new - value_function).max() < tol:
            ok = True
        value_function = v_new.copy()
    ############################
    return value_function, policy

def render_single(env, policy = None, max_steps=100):
    """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
        Environment to play on. Must have nS, nA, and P as
        attributes.
    Policy: np.array of shape [env.nS]
        The action to take at a given state
    """
    # switch env._method() to env.method() for real gym env

    episode_reward = 0
    ob = env._reset()
    for t in range(max_steps):
        env._render()
        time.sleep(0.25)
        a = policy[ob] if (policy is not None) else np.random.randint(0,env.nA)
        ob, rew, done, _ = env._step(a)
        episode_reward += rew
        if done:
            break
    env._render()
    if not done:
        print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
    else:
        print("Episode reward: {} ({} steps)".format(episode_reward, t))
    time.sleep(3.0)


def print_policy(env, pi, v = None):
    d = {0 : 'L' , 1 : 'D', 2 : 'R', 3 : 'U'}
    nrow = env.nrow
    ncol = env.ncol
    for i in range(nrow):
        for j in range(ncol):
            s = i*ncol + j
            print(d[pi[s]], end = '')
            if v is not None:
                print(' ({:#5.3}) '.format(v[s]), end = '')
        print()
    print()

# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below

def init_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--map_name', default = '4x4')
    parser.add_argument('--gamma', default = 0.9, type = float)
    parser.add_argument('--tol', default = 1e-3, type = float)
    parser.add_argument('--slippery', action = 'store_true')
    parser.add_argument('--no_render', action = 'store_true')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = init_args()
    skip_render = args.no_render
    map_name = args.map_name
    is_slippery = args.slippery
    gamma = args.gamma
    tol = args.tol

    # comment/uncomment these lines to switch between deterministic/stochastic environments
    # env = gym.make("Deterministic-4x4-FrozenLake-v0")	
    # env = gym.make("Stochastic-4x4-FrozenLake-v0")

    # using local customized env
    env = FrozenLakeEnv(map_name = map_name, is_slippery = is_slippery)

    print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

    V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=gamma, tol=tol)
    print('# policy evaluations:', len(policy_eval_iter_count), ' : ', policy_eval_iter_count)
    print('Optimal policy:')
    print_policy(env, p_pi, V_pi)
    if not skip_render:
        render_single(env, p_pi, 100)
    
    print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

    V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=gamma, tol=tol)
    print('# value iteration:', n_value_iter)
    print('Optimal policy:')
    print_policy(env, p_vi, V_vi)
    if not skip_render:
        render_single(env, p_vi, 100)
    
    
    
    

