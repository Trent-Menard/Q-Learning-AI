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
        self.is_terminal = {
            "Cool": False,
            "Warm": False,
            "Overheated": True
        }

        # start state
        self.current_state = "Cool"

        # actions
        self.actions = ["Slow", "Fast"]

        # transition model. P(s' | s, a)
        self.transition_model = {
            ("Cool", "Slow"): {"Cool": 1.0, "Warm": 0.0, "Overheated": 0.0},  # P(s' | Cool, Slow)
            ("Cool", "Fast"): {"Cool": 0.5, "Warm": 0.5, "Overheated": 0.0},  # P(s' | Cool, Fast)
            ("Warm", "Slow"): {"Cool": 0.5, "Warm": 0.5, "Overheated": 0.0},  # P(s' | Warm, Slow)
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
            # Randomly select successor state according to transition probabilities
            successor_state = np.random.choice(self.states,
                                               p=[self.transition_model[(self.current_state, action)][sucessor] for
                                                  sucessor in self.states])

            # Update the state
            self.current_state = successor_state

            # Reward
            reward = self.get_reward(action)

            return reward

    def reset_mdp(self):
        self.current_state = "Cool"


class Agent:

    def __init__(self, r_plus, n_e, alpha, gamma=1.0):
        # Call this function to instantiate a Q-learning agent.

        self.r_plus = r_plus  # Optimistic reward value (see exploration function)
        self.n_e = n_e  # Count threshold (see exploration function)
        self.gamma = gamma  # Discount factor
        self.alpha = alpha  # Learning rate

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
            a_prev = None

            # Main Q-learning loop:
            trans_num = 0

            while trans_num <= max_trans:

                # If the environment is in a terminal state, end the trial and start a new one.
                if mdp.current_state == 'Overheated':
                    break

                # Update the agent and get the next action.
                next_action = self.update_and_choose_action(s_prev, a_prev, mdp.current_state, r, self.gamma)

                # Set previous state to this state.
                s_prev = mdp.current_state

                # Set previous action to this action.
                a_prev = next_action

                # Apply the action to the environment and get the resulting reward.
                r = mdp.apply_action(next_action)

                # Increment the transition counter
                trans_num += 1

                # -------------------------------------------------------------------------------------------------#
                # This would probably be a good place to print stuff out for the transition that has now been made:
                # -------------------------------------------------------------------------------------------------#

                print(f"Applied action: '{next_action}' with reward of {r}")

            # Increment the trial counter
            trial_num += 1

        # -------------------------------------------------------------------------------------------------#
        # After everything is finished, be sure to display the resulting policy
        # which can be derived from the Q table.
        # -------------------------------------------------------------------------------------------------#
        self.print_policy()  # You will need to implement this function below.

    def update_and_choose_action(self, s_prev, a_prev, s_curr, r, gamma):
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

            # Increment N[s, a]
            key = (s_prev, a_prev)
            val = self.n_table.get(key)
            self.n_table[key] = val + 1

            # Update Q table
            val = self.q_table.get(key)
            max_q = max(self.q_table.get((s_curr, action)) for action in ["Slow", "Fast"])
            self.q_table[key] = val + self.alpha * (self.n_table[key]) * (r + gamma * max_q - self.q_table[key])

            return

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

        return self.r_plus if n < self.n_e else u

    def print_policy(self):
        """
        # Function that uses self.q_table to print the action each agent should take from each non-terminal state. 
        """
        # ----------------------------------------------------------------------------------------#
        # Implement the function here:                                                            #
        # ----------------------------------------------------------------------------------------#
        #

        for s, a in self.q_table:
            key = (s, a)
            print(key[1])


if __name__ == '__main__':
    # Example usage for how to run your agent:
    my_agent = Agent(10, 3, 1)
    # my_agent.do_q_learning()
    my_agent.do_q_learning(10, 10)
