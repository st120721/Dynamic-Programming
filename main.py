import random
random.seed(0)  # set random set to make sure that produce exactly the same plots

import Investigating_Policy

if __name__ == '__main__':
    Investigating_Policy.plot_trajectories_data(10, 150, 15)
