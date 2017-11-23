import tensorpack
from tensorpack.train.base import Trainer
from tensorpack.input_source import QueueInput, FeedInput
from tensorpack.graph_builder.training import GraphBuilder
from tensorpack.tfutils.tower import TowerContext, get_current_tower_context
from tensorpack.tfutils.gradproc import FilterNoneGrad
import tensorflow as tf

class MultiLossBuilder(GraphBuilder):
    def build(self, input, get_cost_fn, get_opt_fn):
        with TowerContext('', is_training=True) as ctx:
            cost = get_cost_fn(*input.get_input_tensors())
            varlist = ctx.filter_vars_by_vs_name(tf.trainable_variables())
            opt = get_opt_fn()

            train_ops = []
            vl = [[],[]]
            for v in varlist:
                if (v.op.name).startswith('network-1'):
                    vl[1].append(v)
                else:
                    vl[0].append(v)

            for c, o, v in zip(cost, opt, vl):
                if c is None:
                    continue
                print(v)
                grad = o.compute_gradients(c, var_list=v, gate_gradients=False, colocate_gradients_with_ops=True)
                grad = FilterNoneGrad().process(grad)
                train_op = o.apply_gradients(grad, name='min_op_%d' % id(c))
                train_ops.append(train_op)

            #print(train_op)
            #train_op = tf.group(*train_ops, name='train_ops_group')
        return train_ops

class MultiLossTrainer(Trainer):

    def __init__(self, config):
        assert len(config.tower) == 1, \
            "Got nr_tower={}, but doesn't support multigpu!" \
            " Use Sync/AsyncMultiGPUTrainer instead.".format(len(config.tower))

        assert (config.data is not None or config.dataflow is not None) and config.model is not None
        if config.dataflow is None:
            self._input_source = config.data
        else:
            self._input_source = FeedInput(config.dataflow)
            logger.warn("FeedInput is slow (and this is the default of SimpleTrainer). "
                        "Consider QueueInput or other InputSource instead.")
        super(MultiLossTrainer, self).__init__(config)

    def _setup(self):
        cbs = self._input_source.setup(self.model.get_inputs_desc())

        self.train_op = MultiLossBuilder().build(
            self._input_source, self.model.build_graph_get_cost, self.model.get_optimizer)
        self._config.callbacks.extend(cbs)



def MLQueueInputTrainer(config, input_queue=None):
    assert (config.data is not None or config.dataflow is not None) and config.model is not None
    if config.data is not None:
        assert isinstance(config.data, QueueInput), config.data
    else:
        config.data = QueueInput(config.dataflow, input_queue)
    config.dataflow = None
    return MultiLossTrainer(config)

