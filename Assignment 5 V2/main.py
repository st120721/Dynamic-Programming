"""
    This module contains solutions of assignment.
"""
from environment import Environment
import dynamic_programming as dp
import sys
import matplotlib.pyplot as plt
import numpy as np
import time


def main():
    try:
        maze_path = sys.argv[1]
    except:
        sys.exit("Need maze path.")
    else:
        maze_path = "maze.txt"
        e = Environment(maze_path)

        # compare g1 and g2 in VI,PI,OPI
        dp1 = dp.DP(e, gamma=0.9)
        plt.figure(num=1, figsize=(15, 7))
        plt.suptitle("compare g1 and g2 with VI,PI and OPI", fontsize=18, fontweight='bold')
        i = 0
        for k1, f1 in dp1.functions_of_cost_to_go.items():
            for k2, f2 in dp1.functions_of_iterations.items():
                i += 1
                plt.subplot(2, 3, i)
                dp1.plot(k2 + "  " + k1, *f2(f1))
        plt.savefig("1. compare g1 and g2.png")

        # investigate discount factor
        r_lin = np.linspace(0.01, 0.99, num=15, endpoint=True)
        dp2 = dp.DP(e, gamma=0.99)
        reference_policy = dp2.policy_iteration(e.compute_g1)[1]
        differ_list = []
        for r in range(14):
            temp_dp = dp.DP(e, gamma=r_lin[r])
            temp_policy = temp_dp.policy_iteration(e.compute_g1)[1]
            differ = sum([1 for i in range(len(reference_policy)) if reference_policy[i] != temp_policy[i]])
            differ_list.append(differ)
        # plot different entries in the policy from the reference
        plt.figure(num=2)
        plt.suptitle("the different entries in policy under r from the reference", fontsize=12, fontweight='bold')
        plt.plot(r_lin[:14], differ_list)
        plt.xticks(r_lin[:14])
        plt.xlabel("r")
        plt.ylabel("the number of different entries")
        plt.savefig("2. investigate discount factor.png")

        # plot policy for r=0.01
        plt.figure(num=3)
        dp3 = dp.DP(e, gamma=0.01)
        v3, policy3 = dp3.policy_iteration(e.compute_g1)
        dp3.plot("VI  g1  r=0.01", v3, policy3)
        plt.savefig("3. policy of r=0.01.png")

        # compare different algorithms
        runtime = []
        iterations = ["VI", "PI", "OPI"]
        dp4 = dp.DP(e, gamma=0.99)
        for i in iterations:
            start = time.time()
            dp4.functions_of_iterations[i](e.compute_g1)
            end = time.time()
            runtime.append((end - start) * 1000)
        plt.figure(num=4)
        plt.bar(iterations, runtime)
        for i, t in enumerate(runtime):
            plt.text(iterations[i], t + 5, round(t, 2), ha="center", fontsize=10, fontweight='bold')
        plt.title("runtime of VI,PI and OPI", fontweight='bold')
        plt.ylabel("runtime (ms)")
        plt.ylim(0, max(runtime) + 50)
        plt.savefig("4. runtime of VI, PI and OPI.png")

        # investigate m of OPI
        dp5 = dp.DP(e, gamma=0.99)
        runtime_m = []
        for m in range(1, 101, 1):
            print(m)
            start = time.time()
            dp5.functions_of_iterations["OPI"](e.compute_g1, m=m)
            end = time.time()
            runtime_m.append((end - start) * 1000)
        plt.figure(num=5)
        plt.suptitle("runtime of OPI for m(1-100)", fontsize=12, fontweight='bold')
        # plt.plot(list(range(1, 101, 1)), runtime_m)
        plt.bar(list(range(1, 101, 1)), runtime_m)
        plt.xlabel("m")
        plt.xticks(range(10, 105, 10))
        plt.ylabel("runtime (ms)")
        plt.savefig("5. runtime of OPI for different m.png")

        plt.show()


if __name__ == '__main__':
    main()
