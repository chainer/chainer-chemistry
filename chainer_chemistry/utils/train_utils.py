import chainer
from chainer import optimizers, training, Optimizer
from chainer.dataset import convert, Iterator
from chainer.iterators import SerialIterator
from chainer.training import extensions

from chainer_chemistry.training.extensions.auto_print_report import AutoPrintReport  # NOQA


def run_train(model, train, valid=None,
              batch_size=16, epoch=10,
              optimizer=None,
              out='result',
              extensions_list=None,
              device=-1,
              converter=convert.concat_examples,
              use_default_extensions=True,
              resume_path=None):
    """Util function to train chainer's model with StandardUpdater.

    Typical Regression/Classification tasks suffices to use this method to
    train chainer model.

    Args:
        model (chainer.Chain): model to train
        train (dataset or Iterator): training dataset or train iterator
        valid (dataset or Iterator): validation dataset or valid iterator
        batch_size (int): batch size for training
        epoch (int): epoch for training
        optimizer (Optimizer):
        out (str): path for `trainer`'s out directory
        extensions_list (None or list): list of extensions to add to `trainer`
        device (Device): chainer Device
        converter (callable):
        use_default_extensions (bool): If `True`, default extensions are added
            to `trainer`.
        resume_path (None or str): If specified, `trainer` is resumed with this
            serialized file.
    """
    if optimizer is None:
        # Use Adam optimizer as default
        optimizer = optimizers.Adam()
    elif not isinstance(optimizer, Optimizer):
        raise ValueError("[ERROR] optimizer must be instance of Optimizer, "
                         "but passed {}".format(type(Optimizer)))

    optimizer.setup(model)

    if isinstance(train, Iterator):
        train_iter = train
    else:
        # Assume `train` as training dataset, Use SerialIterator as default.
        train_iter = SerialIterator(train, batch_size=batch_size)

    updater = training.StandardUpdater(
        train_iter, optimizer, device=device, converter=converter)
    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    if use_default_extensions:
        if valid is not None:
            if isinstance(valid, Iterator):
                valid_iter = valid
            else:
                # Assume `valid` as validation dataset,
                # Use SerialIterator as default.
                valid_iter = SerialIterator(valid, batch_size=batch_size,
                                            shuffle=False, repeat=False)
            trainer.extend(extensions.Evaluator(
                valid_iter, model, device=device, converter=converter))

        trainer.extend(extensions.LogReport())
        trainer.extend(AutoPrintReport())
        trainer.extend(extensions.ProgressBar(update_interval=10))
        # TODO: consider to include snapshot as default extension.
        # trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))

    if extensions_list is not None:
        for e in extensions_list:
            trainer.extend(e)

    if resume_path:
        chainer.serializers.load_npz(resume_path, trainer)
    trainer.run()

    return
