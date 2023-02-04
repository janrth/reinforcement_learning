Different types of reinforcement learnings will be covered here.

I begin with model based learning and show solutions for FrozenLake using OpenAI gym using value iteration and policy iteration. 

Conceptually in value iteration we do the following:
 - compute the optimal value function first for each state-action pair
 - extract the optimal policy from the optimal value function

While using the policy iteration method we do the following:
 - start with a random policy
 - compute the value function
 - extract a new policy using the value function from the previous step
 - compare the old and the new policy and stop if the difference between them is below a threshold, otherwise continue with another policy (and compute the value function again)
