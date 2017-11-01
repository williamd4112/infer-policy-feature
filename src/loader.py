import tensorpack
from tensorpack.tfutils.sessinit import SessionInit
import tensorflow as tf

def get_model_loader(model, paths, prefixs, mtype):
    return MASaverRestore(model, paths, prefixs, mtype)

class MASaverRestore(SessionInit):

    def __init__(self, model, paths, prefixs=['network-0', 'network-1'], mtype='both', ignore=[]):
        self.model = model
        paths = paths.split(',')
        sep_len = len(paths)
        self.paths = [','.join(paths[i:i+3]) for i in range(0, sep_len, 3)]
        print(self.paths)
        #assert len(self.paths) == 2
        self.prefixs = prefixs
        self.mtype = mtype
        self.ignore = [i if i.endswith(':0') else i + ':0' for i in ignore]

    def _setup_graph(self):
        self.ops = self._get_restore_ops()

    def _run_init(self, sess):
        sess.run(self.ops)


    def _get_restore_ops(self):
        params = tf.trainable_variables()
        vls = [[]]*len(self.prefixs)
        ops = []

        print(len(self.prefixs))
        print(vls)
        for p in params:
            for i, pref in enumerate(self.prefixs):
                if (p.op.name).startswith(pref):
                    vls[i].append(p)

        for path, prefix, vl in zip(self.paths, self.prefixs, vls):
            print("Loading params from {}".format(path))
            reader = tf.train.NewCheckpointReader(path)
            for v in vl:
                if reader.has_tensor(v.op.name):
                    ops.append(v.assign(reader.get_tensor(v.op.name)))
        return tf.group(*ops)


