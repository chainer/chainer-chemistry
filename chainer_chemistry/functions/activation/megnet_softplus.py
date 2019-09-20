from chainer import functions


def megnet_softplus(x):
    """Modified softplus function used by MEGNet

    This function comes from the follwing link.
    https://github.com/materialsvirtuallab/megnet/blob/f91773f0f3fa8402b494638af9ef2ed2807fcba7/megnet/activations.py#L6

    Args:
        x (Variable): Input variable
    Returns:
        output (Variable): Output variable whose shape is same with `x`
    """
    return functions.relu(x) + \
        functions.log(0.5 * functions.exp(-functions.absolute(x)) + 0.5)
