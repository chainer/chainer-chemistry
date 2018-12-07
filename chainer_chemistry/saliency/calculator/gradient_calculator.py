from chainer_saliency.calculator.base_calculator import BaseCalculator


class GradientCalculator(BaseCalculator):

    def __init__(self, model, target_extractor=None, output_extractor=None,
                 eval_fun=None, multiply_target=False, device=None):
        super(GradientCalculator, self).__init__(
            model, target_extractor=target_extractor, output_extractor=output_extractor,
            device=device)
        self.eval_fun = eval_fun or model.__call__
        self.multiply_target = multiply_target

    def _compute_core(self, *inputs):
        self.model.cleargrads()
        outputs = self.eval_fun(*inputs)
        target_var = self.get_target_var(inputs)
        output_var = self.get_output_var(outputs)
        # 1. take sum
        # 2. raise error (default behavior)
        # I think option 1 "take sum" is better, since gradient is calculated
        # automatically independently in that case.
        output_var.backward(retain_grad=True)
        saliency = target_var.grad
        if self.multiply_target:
            saliency *= target_var.data
        outputs = (saliency,)
        return outputs
