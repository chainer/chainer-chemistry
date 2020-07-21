import numpy

from chainer_chemistry.saliency.calculator.gradient_calculator import GradientCalculator  # NOQA


class IntegratedGradientsCalculator(GradientCalculator):

    """Integrated gradient saliency calculator

    Use `compute`, `aggregate` method to calculate saliency.

    See: Mukund Sundararajan, Ankur Taly, and Qiqi Yan (2017).
        Axiomatic attribution for deep networks. PMLR.
        URL http://proceedings.mlr.press/v70/sundararajan17a.html.

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
        baseline (numpy.ndarray or None):
            If `None`, baseline is set as 0.
        steps (int): Number of separation to calculate integrated gradient.
        device (int or None): device id to calculate saliency.
            If `None`, device id is inferred automatically from `model`.
    """
    def __init__(self, model, target_extractor=None, output_extractor=None,
                 eval_fun=None, baseline=None, steps=25, device=None):

        super(IntegratedGradientsCalculator, self).__init__(
            model, target_extractor=target_extractor,
            output_extractor=output_extractor, multiply_target=False,
            eval_fun=eval_fun, device=device)
        self.baseline = baseline or 0.
        self.steps = steps

    def _compute_core(self, *inputs):

        total_grads = 0.
        self.model.cleargrads()
        # Need to forward once to get target_var
        outputs = self.eval_fun(*inputs)  # NOQA
        target_var = self.get_target_var(inputs)
        # output_var = self.get_output_var(outputs)

        base = self.baseline
        diff = target_var.array - base

        for alpha in numpy.linspace(0., 1., self.steps):
            if self.target_extractor is None:
                interpolated_inputs = base + alpha * diff
                inputs[0].array = interpolated_inputs
                total_grads += super(
                    IntegratedGradientsCalculator, self)._compute_core(
                    *inputs)[0]
            else:
                def interpolate_target_var(hook, args, _target_var):
                    interpolated_inputs = base + alpha * diff
                    _target_var.array[:] = interpolated_inputs

                self.target_extractor.add_process(
                    '/saliency/interpolate_target_var', interpolate_target_var)
                total_grads += super(
                    IntegratedGradientsCalculator, self)._compute_core(
                    *inputs)[0]
                self.target_extractor.delete_process(
                    '/saliency/interpolate_target_var')
        saliency = total_grads * diff / self.steps
        return saliency,
