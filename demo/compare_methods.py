import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, 'pycache', 'demo')
lore = __import__('imputlorer')


if __name__ == '__main__':
    # First step
    # Load or generate the data
    X, X0 = lore.generate_data.generateSinusoidalData(size=1000,
                                                      missing_perc=0.2)

    # Second step
    # Run all the imputation methods and collect the data in a dictionary
    standalone_method = ["nocb", "locf", "knn"]
    method = ["simple", "multi"]
    strategy = ["mean", "median", "constant"]

    lore.compare_imputation_methods.compareImputationMethods(X0,
                                                             method=method,
                                                             standalone_method=standalone_method,
                                                             strategy=strategy,
                                                             missing_vals_perc=0.2,
                                                             print_on=True)
