import numpy

from chainer_chemistry.saliency.calculator.gradient_calculator import GradientCalculator  # NOQA


class IntegratedGradientsCalculator(GradientCalculator):

    def __init__(self, model, target_extractor=None, output_extractor=None,
                 eval_fun=None, baseline=None, steps=25):

        super(IntegratedGradientsCalculator, self).__init__(
            model, target_extractor=target_extractor,
            output_extractor=output_extractor, multiply_target=False,
            eval_fun=eval_fun
        )
        self.baseline = baseline or 0.
        self.steps = steps

    def _compute_core(self, *inputs):

        total_grads = 0.
        self.model.cleargrads()
        # Need to forward once to get target_var
        outputs = self.eval_fun(*inputs)
        target_var = self.get_target_var()
        # output_var = self.get_output_var(outputs)

        base = self.baseline
        diff = target_var.array - base

        for alpha in numpy.linspace(0., 1., self.steps):
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
