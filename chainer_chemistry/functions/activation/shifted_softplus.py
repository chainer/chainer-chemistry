import chainer
from chainer import functions


def shifted_softplus(x, beta=1, shift=0.5):
    """shifted softplus function, which holds f(0)=0.

     Args:
        x (Variable): Input variable

    Returns:
        output (Variable): Output variable whose shape is same with `x`
    """
    xp = chainer.cuda.get_array_module(x)
    return functions.softplus(x, beta=beta) + xp.log(shift)
