#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import random
import time
import threading
import multiprocessing
import numpy as np
from tqdm import tqdm
from six.moves import queue

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.stats import *
from tensorpack.utils.utils import get_tqdm_kwargs

def play_one_episode(player, func, verbose=False):
    def f(s):
        spc = player.get_action_space()
        output = func([[s], [[0.06806452]*13], [[1201]*13]])
        act = [np.argmax(o) for o in output]
        if random.random() < 0.001:
            act = [spc.sample(), spc.sample()]
        return act
    r, s = player.play_one_episode(f, ['score', 'scorer'])
    return r[0][0], s


def play_model(cfg, player):
    predfunc = OfflinePredictor(cfg)
    while True:
        score = play_one_episode(player, predfunc)


def eval_with_funcs(predictors, nr_eval, get_player_fn):
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        score = play_one_episode(player, self.func)
                        # print("Score, ", score)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(2.0)  # avoid simulator bugs
    stat = StatCounter()
    ep = 0
    scorer = []
    try:
        for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
            ep += 1
            r, s = q.get()
            stat.feed(r)
            if ep % 250 == 0:
                print("({}) avg: {}, max {}".format(ep, stat.average, stat.max))
            if len(s) > 0:
                scorer.append(s[0])
            elif r == 0:
                scorer.append(5)
        logger.info("Waiting for all the workers to finish the last run...")
        for k in threads:
            k.stop()
        for k in threads:
            k.join()
        while q.qsize():
            r = q.get()
            stat.feed(r)
    except:
        logger.exception("Eval")
    finally:
        scorer = np.array(scorer)
        if len(scorer) > 0:
            print("scorer 0: {}".format(np.count_nonzero(scorer==0)))
            print("scorer 1: {}".format(np.count_nonzero(scorer==1)))
            print("times of draw: {}".format(np.count_nonzero(scorer==5)))
        if stat.count > 0:
            return (stat.average, stat.max)
        return (0, 0)


def eval_model_multithread(cfg, nr_eval, get_player_fn):
    nr_eval = 100000
    player = get_player_fn()
    '''
    predfunc = OfflinePredictor(cfg)
    scores = []
    for ep in range(nr_eval):
        scores.append(play_one_episode(player, predfunc))
        scores_ = np.asarray(scores)
        print("Episode [%d]:" % ep)
        print("Player 0 - Mean: %f, Max: %f, Min %f" % (np.mean(scores_[:, 0, 0]), np.max(scores_[:, 0, 0]), np.min(scores_[:, 0, 0])))
        print("Player 1 - Mean: %f, Max: %f, Min %f" % (np.mean(scores_[:, 0, 1]), np.max(scores_[:, 0, 1]), np.min(scores_[:, 0, 1])))
    scores = np.array(scores)
    print("Evaluation over!")
    print("Player 0 - Mean: %f, Max: %f, Min %f" % (np.mean(scores[:, 0, 0]), np.max(scores[:, 0, 0]), np.min(scores[:, 0, 0])))
    print("Player 1 - Mean: %f, Max: %f, Min %f" % (np.mean(scores[:, 0, 1]), np.max(scores[:, 0, 1]), np.min(scores[:, 0, 1])))
    '''
    func = OfflinePredictor(cfg)
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    mean, max = eval_with_funcs([func] * NR_PROC, nr_eval, get_player_fn)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))

class Evaluator(PeriodicTrigger):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)


def play_n_episodes(player, predfunc, nr):
    logger.info("Start evaluation: ")
    for k in range(nr):
        if k != 0:
            player.restart_episode()
        score = play_one_episode(player, predfunc)
        print("{}/{}, score=", k, nr, score)
