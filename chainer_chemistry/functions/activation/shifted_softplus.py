import chainer
from chainer import functions


def shifted_softplus(x, beta=1, shift=0.5, threshold=20):
    """shifted softplus function, which holds f(0)=0.

     Args:
        x (Variable): Input variable
        beta (float): Parameter :math:`\\beta`.
        shift (float): Shift Parameter
        threshold (float): threshold to avoid overflow

    Returns:
        output (Variable): Output variable whose shape is same with `x`
    """
    xp = chainer.cuda.get_array_module(x)
    cond = chainer.as_variable(x).array > threshold
    x = functions.where(cond, x,
                        functions.softplus(x, beta=beta))
    x += xp.log(shift)
    return x
