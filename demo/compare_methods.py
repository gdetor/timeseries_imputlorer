# This demo file shows how to use compare different time series data
# imputation methods.
# Copyright (C) 2024 Georgios Is. Detorakis
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import os
import sys

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.pycache_prefix = os.path.join(root_dir, '__pycache__', 'demo')
lore = __import__('imputlorer')


if __name__ == '__main__':
    # First step
    # Load or generate the data
    X, X0 = lore.generate_data.generateSinusoidalData(size=100,
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
