from chainer import reporter
from chainer_chemistry.models.prediction.classifier import Classifier


class NodeClassifier(Classifier):
    """A simple node classifier model."""

    def __call__(self, data, train_mask, valid_mask, *args, **kwargs):
        """Computes the loss value for an input and label pair."""
        self.metrics = None
        self.y = self.predictor(data)
        # Support for padding pattern
        if self.y.ndim == 3:
            assert self.y.shape[0] == 1
            self.y = self.y[0]
        self.train_loss = self.lossfun(self.y[train_mask], data.y[train_mask])
        self.valid_loss = self.lossfun(self.y[valid_mask], data.y[valid_mask])
        reporter.report(
            {'loss(train)': self._convert_to_scalar(self.train_loss)}, self)
        reporter.report(
            {'loss(valid)': self._convert_to_scalar(self.valid_loss)}, self)
        if self.compute_metrics:
            # Note: self.accuracy is `dict`, which is different from original
            # chainer implementation
            self.train_metrics = {key + "(train)":
                                  self._convert_to_scalar(
                                      value(self.y[train_mask],
                                            data.y[train_mask]))
                                  for key, value in self.metrics_fun.items()}
            self.valid_metrics = {key + "(valid)":
                                  self._convert_to_scalar(
                                      value(self.y[valid_mask],
                                            data.y[valid_mask]))
                                  for key, value in self.metrics_fun.items()}
            reporter.report(self.train_metrics, self)
            reporter.report(self.valid_metrics, self)
        return self.train_loss
