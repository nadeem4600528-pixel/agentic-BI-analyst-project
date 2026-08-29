
import pandas as pd
import numpy as np
from profiling.correlation import CorrelationProfiler

def test_correlation():
    data = {
        "A": [1, 2, 3, 4, 5],
        "B": [5, 4, 3, 2, 1],
        "C": [1, 1, 0, 0, 1]
    }
    df = pd.DataFrame(data)
    profiler = CorrelationProfiler(df)
    
    print("Running Pearson...")
    print(profiler.pearson_correlation())
    
    print("\nRunning Spearman...")
    print(profiler.spearman_correlation())
    
    print("\nRunning Kendall Tau...")
    print(profiler.kendall_tau())
    
    print("\nRunning Cramers V...")
    print(profiler.cramers_v())
    
    print("\nRunning Theils U...")
    print(profiler.theils_u())
    
    print("\nRunning Mutual Information...")
    print(profiler.mutual_information())

if __name__ == "__main__":
    test_correlation()
