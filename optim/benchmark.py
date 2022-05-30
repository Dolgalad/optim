"""Benchmarking optimization algorithm
"""
import time
import numpy as np
import pandas as pd

from optim.test_function import test_functions

def get_test_functions_for_dim(dim):
    ans = [tf(n=dim) for tf in test_functions]
    return [tf.__class__ for tf in ans if dim==tf.n]

def benchmark(optimizers, initial_condition):
    dim = initial_condition.shape[-1]
    columns = ["Optimizer","Test function","runtime","l2_error","optimizer"]
    data = {k:[] for k in columns}
    results = pd.DataFrame(data = data, columns=columns)
    if isinstance(optimizers, list):
        for optimizer in optimizers:
            results = results.append(benchmark(optimizer, initial_condition))
        return results
    for tf in get_test_functions_for_dim(dim):
        # instantiate the test function
        F = tf(n=dim)
        # solve the optimization problem with the algorithm
        t0 = time.time()
        x_solution = optimizers.optimize(F, x0=initial_condition)
        t0 = time.time() - t0
        # get distance to optimal solution
        if len(F.x_star.shape)==1:
            l2_norm = np.linalg.norm(F.x_star - x_solution)
        else:
            l2_norm = np.linalg.norm(F.x_star - x_solution, axis=1)
            l2_norm = np.min(l2_norm)
        results = results.append({"Optimizer":optimizers.__class__.__name__,
            "Test function":tf.__name__,
            "runtime":t0,
            "l2_error":l2_norm,
            "optimizer":optimizers,
            "iterations":optimizers.it}, ignore_index=True)
    #results["iterations"]=[o.it for o in results["optimizer"]]
    return results

import matplotlib.pyplot as plt
import seaborn as sns
def test_function_result_bar_plot(results, title="Test function optimization", **seaborn_kwargs):
    ax=sns.barplot(x="l2_error", y="Test function",orient="h",data=results,**seaborn_kwargs)
    #ax.set_xticklabels(labels,rotation=45)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("1 / ||\hat{x} - x^*||")
    return ax

def iteration_bar_plot(results, title="Test function optimization",**seaborn_kwargs):
    ax=sns.barplot(x="iterations", y="Test function",orient="h",data=results,**seaborn_kwargs)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("iterations")
    return ax

def runtime_bar_plot(results, title="Test function optimization", **seaborn_kwargs):
    ax=sns.barplot(x="runtime", y="Test function",orient="h",data=results, **seaborn_kwargs)
    #ax.set_xticklabels(labels,rotation=45)
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel("runtime (ms)")
    return ax



if __name__=="__main__":
    from optim.utils import print_dict

    # test the RandomOptimizer
    from optim.blackbox.random import RandomOptimizer
    from optim.blackbox.pattern import NelderMead
    from optim.blackbox.cmaes import CMAES

    optimizer = RandomOptimizer()
    x0 = np.random.random_sample(5)
    results = benchmark(optimizer, x0)
    
    # create a first simple figure
    #print_dict(results)

    #ax = test_function_result_bar_plot(results)
    #plt.show()
    #ax = iteration_bar_plot(results)
    #plt.show()
    #ax = runtime_bar_plot(results)
    #plt.show()

    optimizers = [RandomOptimizer(), CMAES(), NelderMead()]
    results = benchmark(optimizers, x0)
    print(results)

    ax = test_function_result_bar_plot(results, hue="Optimizer")
    plt.show()
    ax = iteration_bar_plot(results, hue="Optimizer")
    plt.show()
    ax = runtime_bar_plot(results, hue="Optimizer")
    plt.show()





