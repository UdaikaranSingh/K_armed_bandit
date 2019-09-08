import numpy as np


def create_model(k, max_sd, min_mean, max_mean):
    # k ~ # of bandits
    # max_sd ~ maximum standard deviation of bandit (minimum SD is 0)
    # min/max_mean ~ range for mean of normal distribution
    
    means = np.array(list(range(min_mean,max_mean + 1)))
    sd_s = np.array([x/100 for x in list(range(0,max_sd * 100))])
    sample_means = np.random.choice(means, size = k)
    sample_sds = np.random.choice(sd_s, size = k)
    
    model = list(zip(sample_means, sample_sds))
    
    return model

def simulation_e_greedy(ground_truth, policy, epsilon, n, k, step_size, c):
	# ground_truth ~ this is the ground truth of the distributions that stores the datapoints
	# policy ~ means of bandits
	# epsilon ~ 
	# n ~ number of total steps
	# k ~ k parameter
	# step_size ~ alpha value

	counts = np.zeros(k)
	total_reward = 0
	choices = []

	for t in range(n):

		path = np.random.choice(a = ["greedy", "exploration"], size = 1, p = [1 - epsilon, epsilon])[0]

		if (path == "greedy"):
			choice = np.argmax(policy)
		else:
			choice = np.random.choice(range(k))

		counts[choice] += 1
		choices.append(choice)
		bandit_choice = ground_truth[choice]
		bandit_mean, bandit_sd = bandit_choice
		reward = np.random.normal(bandit_mean, bandit_sd)

		policy[choice] = (policy[choice] + (1 / counts[choice]) * (reward - policy[choice]) 
                          + c * np.sqrt(np.log(t + 1)/ counts[choice]))
		total_reward += reward

	return (counts, choices, policy, total_reward)
