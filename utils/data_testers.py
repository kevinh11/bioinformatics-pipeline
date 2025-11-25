from scipy.stats import ranksums
import numpy as np

def is_train_test_difference_significant(x_train, x_test, alpha=0.05):
    """
    Perform a Wilcoxon rank-sum test to determine if there's a statistically significant
    difference between the distributions of training and test sets.
    
    Args:
        x_train: Array-like, training set samples
        x_test: Array-like, test set samples
        alpha: Significance level (default: 0.05)
        
    Returns:
        tuple: (is_significant, p_value)
            - is_significant (bool): True if difference is statistically significant
            - p_value (float): The p-value of the test
    """
    # Convert inputs to numpy arrays if they aren't already
    x_train = np.asarray(x_train)
    x_test = np.asarray(x_test)
    
    # Flatten arrays if they are multi-dimensional
    if x_train.ndim > 1:
        x_train = x_train.ravel()
    if x_test.ndim > 1:
        x_test = x_test.ravel()
    
    # Perform the Wilcoxon rank-sum test
    stat, p_value = ranksums(x_train, x_test)
    
    # Check if the difference is statistically significant
    is_significant = p_value < alpha

    
    
    return is_significant, p_value
