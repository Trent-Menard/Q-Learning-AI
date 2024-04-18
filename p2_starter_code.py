# -*- coding: utf-8 -*-
"""
# Starter code for project 2. 
"""

import numpy as np

class RacecarMDP:
    """
    # This class defines the race MDP.
    # Your agent will need to interact with an instantiation of this class as part of your implementation of Q-learning.
    # Note that your agent should not access anything from this class except through the apply_action() function. 
    """
    
    def __init__(self):
        # states
        self.states = ["Cool", "Warm", "Overheated"]

        # terminal_states
        self.is_terminal = {"Cool": False,
                            "Warm": False,
                            "Overheated": True}

        # start state
        self.current_state = "Cool"

        # actions
        self.actions = ["Slow", "Fast"]

        # transition model. P(s' | s, a)
        self.transition_model = {
            ("Cool", "Slow"): {"Cool": 1.0, "Warm": 0.0, "Overheated": 0.0}, # P(s' | Cool, Slow)
            ("Cool", "Fast"): {"Cool": 0.5, "Warm": 0.5, "Overheated": 0.0}, # P(s' | Cool, Fast)
            ("Warm", "Slow"): {"Cool": 0.5, "Warm": 0.5, "Overheated": 0.0}, # P(s' | Warm, Slow)
            ("Warm", "Fast"): {"Cool": 0.0, "Warm": 0.0, "Overheated": 1.0}  # P(s' | Warm, Fast)
        }


    
    def get_reward(self, action):
        # Defines the reward function.
        if self.current_state != "Overheated":
            if action == "Fast":
                return 20
            elif action == "Slow":
                return 10

        else:
            # Current state is Overheated
            return -50


    def apply_action(self, action):
        """
        This function updates the environment state in response to some action taken.
        It returns the new state and the reward received. 
        """
        if self.current_state == "Overheated":
            print("your racecar has overheated :(")
            return 0, 0

        else:
            
            # VERIFY IS self.states & not mdp.states

            # Randomly select successor state according to transition probabilities
            successor_state = np.random.choice(self.states,
                                               p=[self.transition_model[(self.current_state, action)][sucessor] for sucessor in self.states])

            # Update the state
            self.current_state = successor_state

            # Reward
            reward = self.get_reward(action)

            return reward


    def reset_mdp(self):
        self.current_state = "Cool"
        

class Agent:
    
    def __init__(self, r_plus, n_e, gamma=1.0):
        """
        # Call this function to instantiate a Q-learning agent. Examples:
        # >> my_agent = Agent(100, 5)
        # >> my_agent_w_discounting = Agent(100, 5, gamma=0.8)
        #
        # For this project, it is fine to just leave gamma on 1.0 (but you should explain what that means
        # in the tutorial.
        #
        """
        
        self.r_plus = r_plus  # Optimistic reward value (see exploration function) 
        self.n_e = n_e        # Count threshold (see exploration function)
        self.gamma = gamma    # Discount factor

        # ----------------------------------------------------------------------------------------#
        # Initialize the following. For the tables, there are lots of ways you might implement    #
        # them, but I would consider using dictionaries or 2-dimensional lists.                   #
        # It is safe to assume that the agent knows the full space of states and actions that are # 
        # possible.                                                                               #
        # For gamma, r_plus, and n_e: you can try any values you like and see how it goes.        #
        # ----------------------------------------------------------------------------------------#
        #
        # self.q_table = ???      # Note that this is referred to as Q in the algorithm
        # self.n_table = ???      # Note that this is referred to as N_sa in the algorithm

        self.q_table = {
            ("Cool", "Slow"): 0,
            ("Cool", "Fast"): 0,
            ("Warm", "Slow"): 0,
            ("Warm", "Fast"): 0,
            ("Overheated", "Slow"): 0,
            ("Overheated", "Fast"): 0
        }

        self.n_table = self.q_table.copy()


    def do_q_learning(self, max_trials=100, max_trans=100):
        """
        # Outer function for Q-learning. The main Q-learning algorithm will need to be implemented in
        # the function update_and_choose_action(). You will also need to implement 
        """
        
        mdp = RacecarMDP()
        trial_num = 1

        while trial_num <= max_trials:

            # Set up the MDP and initialize the reward to 0
            mdp.reset_mdp()
            r = 0
            s_prev = None

            # Main Q-learning loop:
            trans_num = 0
            
            while trans_num <= max_trans:

                # Check to see if the environment is in a terminal state. If so, we have to end the trial and start a new one.
                if mdp.current_state == 'Overheated':
                    break

                # Update the agent and get the next action.
                next_action = self.update_and_choose_action(s_prev, mdp.current_state, r, self.gamma)

                # Apply the action to the environment and get the resulting reward.
                r = mdp.apply_action(next_action)

                # Increment the transition counter
                trans_num += 1

                # -------------------------------------------------------------------------------------------------#
                # This would probbaly be a good place to print stuff out for the transition that has now been made:
                # -------------------------------------------------------------------------------------------------#
                #
                #
                #
                #

            # Increment the trial counter
            trial_num += 1

        # -------------------------------------------------------------------------------------------------#
        # After everything is finished, be sure to display the resulting policy
        # which can be derived from the Q table.
        # -------------------------------------------------------------------------------------------------#
        self.print_policy()  # You will need to implement this function below.
        

    def update_and_choose_action(self, s_prev, s_curr, r, gamma):
        """
        # This function should be a translation of the Q-learning algorithm from the project document.
        # It should take the current state and current reward as input (the percept) and output the next
        # action the agent should take.
        #
        # s_curr: the current state. s' in the algorithm.
        # r     : the transition reward just received
        # gamma : the discount factor
        """

        if s_prev is not None:
            
            # ----------------------------------------------------------------------------------------#
            # Implement what should normally happen here using the Q-Learning-Agent algorithm:        #
            # ----------------------------------------------------------------------------------------#
            #
            # 
            #
            # 

            n_key = (s_prev, s_curr)
            # n_val = 
            # self.n_table[key] = val + 1

            exit()
            
        else:
            # Otherwise, if s_curr is the very first state in a trial (i.e., there have not been any transitions yet),
            # then the best we can do is choose an action at random:
            return np.random.choice(['Slow', 'Fast']) 
        


    def f(self, u, n):
        """
        # Exploration function that you will need to implement.
        #
        # u: the utility value
        # n: the number of times the state-action pair has been tried
        """

        # ----------------------------------------------------------------------------------------#
        # Implement the exploration function here:                                                #
        # ----------------------------------------------------------------------------------------#
        #


    def print_policy(self):
        """
        # Function that uses self.q_table to print the action each agent should take from each non-terminal state. 
        """
        # ----------------------------------------------------------------------------------------#
        # Implement the function here:                                                            #
        # ----------------------------------------------------------------------------------------#
        #
        

if __name__ == '__main__':
    # Example usage for how to run your agent:
    my_agent = Agent(10, 3)
    my_agent.do_q_learning()