import chainer  # NOQA
from chainer import functions
from chainer_chemistry.saliency.calculator.base_calculator import BaseCalculator  # NOQA


class GradientCalculator(BaseCalculator):

    """Gradient saliency calculator

    Use `compute`, `aggregate` method to calculate saliency.

    See: Dumitru Erhan, Yoshua Bengio, Aaron Courville, Pascal Vincent (2009).
        Visualizing Higher-Layer Features of a Deep Network.

    See: Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman.
        Deep inside convolutional networks: Visualising image classication
        models and saliency maps.
        `arXiv:1312.6034 <https://arxiv.org/abs/1312.6034>`_

    Args:
        model (chainer.Chain): target model to calculate saliency.
        target_extractor (VariableMonitorLinkHook or None):
            It determines `target_var`, target variable to calculate saliency.
            If `None`, first argument of input to the model is treated as
            `target_var`.
        output_extractor (VariableMonitorLinkHook or None):
            It determines `output_var`, output variable to calculate saliency.
            If `None`, output of the model is treated as `output_var`.
        eval_fun (callable): If
        multiply_target (bool):
            If `False`, return value is `target_var.grad`.
            If `True`,  return value is `target_var.grad * target_var`.
        device (int or None): device id to calculate saliency.
            If `None`, device id is inferred automatically from `model`.
    """

    def __init__(self, model, target_extractor=None, output_extractor=None,
                 eval_fun=None, multiply_target=False, device=None):
        super(GradientCalculator, self).__init__(
            model, target_extractor=target_extractor,
            output_extractor=output_extractor, device=device)
        self.eval_fun = eval_fun or model.__call__
        self.multiply_target = multiply_target

    def _compute_core(self, *inputs):
        self.model.cleargrads()
        outputs = self.eval_fun(*inputs)
        target_var = self.get_target_var(inputs)
        target_var.grad = None  # Need to reset grad beforehand of backward.
        output_var = self.get_output_var(outputs)

        # --- type check for output_var ---
        if output_var.size != 1:
            self.logger.warning(
                'output_var.size is not 1, calculate scalar value. '
                'functions.sum is applied.')
            output_var = functions.sum(output_var)

        output_var.backward(retain_grad=True)
        saliency = target_var.grad
        if self.multiply_target:
            saliency *= target_var.data
        outputs = (saliency,)
        return outputs
