"""
Copyright 2020 Sensetime X-lab. All Rights Reserved

Main Function:
    1. base class for model learning
"""
import os
from abc import ABC, abstractmethod
from typing import Any, Union, Callable
from easydict import EasyDict
import torch

from torch.utils.data._utils.collate import default_collate
from distar.ctools.torch_utils import build_checkpoint_helper, CountVar, auto_checkpoint, build_log_buffer
from distar.ctools.utils import build_logger, dist_init, EasyTimer, dist_finalize, pretty_print, read_config, DistModule
from distar.ctools.utils import deep_merge_dicts
from distar.ctools.worker.learner.learner_hook import build_learner_hook_by_cfg, add_learner_hook, merge_hooks, LearnerHook
from distar.ctools.torch_utils.grad_clip import build_grad_clip
from distar.ctools.torch_utils.lr_scheduler_util import GradualWarmupScheduler

default_config = read_config(os.path.join(os.path.dirname(__file__), "base_learner_default_config.yaml"))


class BaseLearner(ABC):
    r"""
    Overview:
        base class for model learning(SL/RL), which is able to multi-GPU learning
    Interface:
        __init__, register_stats, run, close, call_hook, info, save_checkpoint, launch
    Property:
        last_iter, optimizer, lr_scheduler, computation_graph, agent, log_buffer, record,
        load_path, save_path, checkpoint_manager, name, rank, tb_logger, use_distributed
    """

    _name = "BaseLearner"  # override this variable for sub-class learner

    def __init__(self, cfg: EasyDict, method="single_node", init_method=None, rank=0, world_size=1) -> None:
        """
        Overview:
            initialization method, load config setting and call ``_init`` for actual initialization,
            set the communication mode to `single_machine` or `flask_fs`.
        Arguments:
            - cfg (:obj:`EasyDict`): learner config, you can view `cfg <../../../configuration/index.html>`_ for ref.
        Notes:
            if you want to debug in sync CUDA mode, please use the following line code in the beginning of ``__init__``.

            .. code:: python

                os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # for debug async CUDA
        """
        self._whole_cfg = deep_merge_dicts(default_config, cfg)
        self._load_path = self._whole_cfg.learner.load_path
        self._experiment_name = self._whole_cfg.common.experiment_name
        self._use_cuda = self._whole_cfg.learner.use_cuda and torch.cuda.is_available()
        self._use_distributed = self._whole_cfg.learner.use_distributed and self._use_cuda
        if self._use_distributed:
            self._rank, self._world_size = dist_init(method=method, init_method=init_method, rank=rank, world_size=world_size)
        else:
            self._rank, self._world_size = 0, 1
        self._device = torch.cuda.current_device() if self._use_cuda else 'cpu'
        self._default_max_iterations = self._whole_cfg.learner.max_iterations
        self._timer = EasyTimer(self._use_cuda)
        # checkpoint helper
        self._checkpointer_manager = build_checkpoint_helper(self._whole_cfg)
        self._hooks = {'before_run': [], 'before_iter': [], 'after_iter': [], 'after_run': []}
        self._collate_fn = default_collate
        self._init_model()
        self._setup_dataloader()
        self._setup_optimizer()
        self._setup_loss()
        self._setup_wrapper()
        self._setup_hook()

        # logger
        self._logger, self._tb_logger, self._record = build_logger(self._whole_cfg, rank=self._rank)
        self._log_buffer = build_log_buffer()

        self._last_iter = CountVar(init_val=0)
        if self._rank == 0:
            self.register_stats()
            self.info(
                pretty_print(
                    {
                        "config": self._whole_cfg,
                    },
                    direct_print=False
                )
            )

    def _init_model(self):
        self._setup_model()
        if self._use_cuda:
            self._model = self._model.to(device=self._device)
        if self.use_distributed:
            self._model = DistModule(self._model)
        self._grad_clip = build_grad_clip(self._whole_cfg.learner.grad_clip)

    def _setup_hook(self) -> None:
        """
        Overview:
            Setup hook for base_learner. Hook is the way to implement actual functions in base_learner.
            You can reference learner_hook.py
        """
        if hasattr(self, '_hooks'):
            self._hooks = merge_hooks(self._hooks, build_learner_hook_by_cfg(self._whole_cfg.learner.hook))
        else:
            self._hooks = build_learner_hook_by_cfg(self._whole_cfg.learner.hook)

    def _setup_wrapper(self) -> None:
        """
        Overview:
            Setup time_wrapper to get data_time and train_time
        """
        self._wrapper_timer = EasyTimer(cuda=self._use_cuda)
        self._get_iter_data = self.time_wrapper(self._get_iter_data, 'data_time')
        self._train = self.time_wrapper(self._train, 'train_time')

    def time_wrapper(self, fn: Callable, name: str):
        """
        Overview:
            Wrap a function and measure the time it used
        Arguments:
            - fn (:obj:`Callable`): function to be time_wrapped
            - name (:obj:`str`): name to be registered in log_buffer
        """

        def wrapper(*args, **kwargs) -> Any:
            with self._wrapper_timer:
                ret = fn(*args, **kwargs)
            self._log_buffer[name] = self._wrapper_timer.value
            return ret

        return wrapper

    def _setup_dataloader(self) -> None:
        raise not NotImplementedError

    def _setup_model(self) -> None:
        """
        Overview:
            Setup learner's runtime agent, agent is the subclass instance of `BaseAgent`.
            There may be more than one agent.
        Note:
            `agent` is the wrapped `model`, it can be wrapped with different plugins to satisfy
            different runtime usages (e.g. actor and learner would use the model differently)
        """
        raise NotImplementedError

    def _setup_loss(self) -> None:
        """
        Overview:
            Setup computation_graph, which uses procssed data and agent to get an optimization
            computation graph.
        """
        raise NotImplementedError

    def _setup_optimizer(self) -> None:
        """
        Overview:
            Setup learner's optimizer and lr_scheduler
        """
        lr_decay = self._whole_cfg.learner.get('lr_decay', 1.)
        lr_decay_interval = int(self._whole_cfg.learner.get('lr_decay_interval', 1e20))
        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._whole_cfg.learner.learning_rate,
            weight_decay=self._whole_cfg.learner.weight_decay
        )
        # warm_up
        if self._whole_cfg.learner.use_warmup:
            lr_decay = self._whole_cfg.learner.get('lr_decay', 0.9)
            lr_decay_interval = int(self._whole_cfg.learner.get('lr_decay_interval', 10000))
            multiplier = self._whole_cfg.learner.get('multiplier', 1)
            warm_up_steps = self._whole_cfg.learner.get('warm_up_steps', 10000)
            self._after_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=list(
                range(0, lr_decay_interval * 40, lr_decay_interval))[1:], gamma=lr_decay)
            self._lr_scheduler = GradualWarmupScheduler(optimizer=self._optimizer, multiplier=multiplier,
                                                       total_epoch=warm_up_steps,
                                                       after_scheduler=self._after_lr_scheduler)
        else:
            self._lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer, milestones=list(range(0, lr_decay_interval * 20, lr_decay_interval))[1:], gamma=lr_decay)

    def _get_iter_data(self):
        return next(self._dataloader)

    def _train(self, data: Any) -> None:
        """
        Overview:
            Train the input data for 1 iteration, called in ``run`` which involves:

                - forward
                - backward
                - sync grad (if in distributed mode)
                - parameter update
        Arguments:
            - data (:obj:`Any`): data used for training
        """
        with self._timer:
            data = self._model(data)
            log_vars = self._loss.compute_loss(data)
            loss = log_vars['total_loss']
            print('loss',loss)
        self._log_buffer['forward_time'] = self._timer.value

        with self._timer:
            self._optimizer.zero_grad()
            loss.backward()
            if self._use_distributed:
                self._model.sync_gradients()
            gradient = self._grad_clip.apply(self._model.parameters())
            self._optimizer.step()
        self._log_buffer['gradient'] = gradient
        self._log_buffer['backward_time'] = self._timer.value
        self._log_buffer.update(log_vars)

    def register_stats(self) -> None:
        """
        Overview:
            register some basic attributes to record & tb_logger(e.g.: cur_lr, data_time, train_time),
            register the attributes related to computation_graph to record & tb_logger.
        """
        self._record.register_var('cur_lr')
        self._record.register_var('data_time')
        self._record.register_var('train_time')
        self._record.register_var('forward_time')
        self._record.register_var('backward_time')
        self._record.register_var('gradient')

        self._tb_logger.register_var('cur_lr')
        self._tb_logger.register_var('data_time')
        self._tb_logger.register_var('train_time')
        self._tb_logger.register_var('forward_time')
        self._tb_logger.register_var('backward_time')
        self._tb_logger.register_var('gradient')

        if hasattr(self._loss, 'register_stats'):
            self._loss.register_stats(self._record, self._tb_logger)

    def register_hook(self, hook: LearnerHook) -> None:
        """
        Overview:
            Add a new hook to learner.
        Arguments:
            - hook (:obj:`LearnerHook`): the hook to be added to learner
        """
        add_learner_hook(self._hooks, hook)

    @auto_checkpoint
    def run(self, max_iterations: Union[int, None] = None) -> None:
        """
        Overview:
            Run the learner.
            For each iteration, learner will get training data and train.
            Learner will call hooks at four fixed positions(before_run, before_iter, after_iter, after_run).
        Arguments:
            - max_iterations (:obj:`int`): the max run iteration, if None then set to default_max_iterations
        """
        if max_iterations is None:
            max_iterations = self._default_max_iterations
        # before run hook
        self.call_hook('before_run')

        for i in range(max_iterations):
            data = self._get_iter_data()
            # print('=123',i, max_iterations)
            # before iter hook
            self.call_hook('before_iter')
            self._train(data)
            # after iter hook
            self.call_hook('after_iter')
            self._last_iter.add(1)

        # after run hook
        self.call_hook('after_run')

    def close(self) -> None:
        """
        Overview:
            Close the related resources, such as dist_finalize when use_distributed
        """
        if self._use_distributed:
            dist_finalize()

    def call_hook(self, name: str) -> None:
        """
        Overview:
            Call the corresponding hook plugins according to name
        Arguments:
            - name (:obj:`str`): hooks in which position to call, \
                should be in ['before_run', 'after_run', 'before_iter', 'after_iter']
        """
        for hook in self._hooks[name]:
            hook(self)

    def info(self, s: str) -> None:
        """
        Overview:
            Log string info by ``self._logger.info``
        Arguments:
            - s (:obj:`str`): the message to add into the logger
        """
        self._logger.info(s)

    def save_checkpoint(self) -> None:
        """
        Overview:
            Automatically save checkpoints.
            Directly call ``save_ckpt_after_run`` hook instead of calling ``call_hook`` function.
        Note:
            This method is called by `auto_checkpoint` function in `checkpoint_helper.py`,
            designed for saving checkpoint whenever an exception raises.
        """
        names = [h.name for h in self._hooks['after_run']]
        assert 'save_ckpt_after_run' in names
        idx = names.index('save_ckpt_after_run')
        self._hooks['after_run'][idx](self)

    @property
    def last_iter(self) -> CountVar:
        return self._last_iter

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    @property
    def lr_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        return self._lr_scheduler

    @property
    def loss(self) -> Any:
        return self._loss

    @property
    def model(self):
        return self._model

    @property
    def log_buffer(self) -> dict:  # LogDict
        return self._log_buffer

    @log_buffer.setter
    def log_buffer(self, _log_buffer: dict) -> None:
        self._log_buffer = _log_buffer

    @property
    def record(self) -> 'VariableRecord':  # noqa
        return self._record

    @property
    def load_path(self) -> str:
        return self._load_path

    @load_path.setter
    def load_path(self, _load_path: str) -> None:
        self._load_path = _load_path

    @property
    def experiment_name(self) -> str:
        return self._experiment_name

    @property
    def checkpoint_manager(self) -> Any:
        return self._checkpointer_manager

    @property
    def name(self) -> str:
        return self._name + str(id(self))

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    @property
    def tb_logger(self) -> 'TensorBoardLogger':  # noqa
        return self._tb_logger

    @property
    def use_distributed(self) -> bool:
        return self._use_distributed

