from chainer import functions


def improved_softplus(x):
    """
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, 
    the implementation aims at avoiding overflow

    Args:
        x: (.....) input
    Returns:
         (.....) output
    """
    return functions.relu(x) + \
        functions.log(0.5 * functions.exp(-functions.absolute(x)) + 0.5)
