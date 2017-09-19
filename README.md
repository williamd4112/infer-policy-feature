# Introduction
This repository is the tensorflow implementations of the paper DPIQN: Deep Policy Inference Q-Network.
# Requirments
- Python 3.4+
- [pygame-soccer](https://github.com/ebola777/pygame-soccer)
- tensorflow 1.2.1
- [tensorpack](https://github.com/ppwwyyxx/tensorpack)

# Training
Simply enter the following command to train a DPIQN agent:
```
 python src/train_dpiqn.py
```
 
The following arguments can help you customize your own training arguments:
 ```
  --gpu GPU             comma separated list of GPU(s) to use.
  --load LOAD           load model
  --log LOG             train log dir
  --task                task to perform {play, eval, train}
  --algo                algorithm for computing Q-value {DQN, Double, Dueling}
  --mode                specify ai mode in env (can be list) {offensive, defensive}
  --mt_mode             multi-task setting {coop-only,opponent-only,all}
  --mt                  use 2v2 env
  --skip SKIP           act repeat
  --hist_len            hist len
  --batch_size          batch size (default: 32)
  --lr LR               init lr value (default: 1e-3)
  --rnn RNN             use rnn (DRPIQN)
  --lr_sched LR_SCHED   lr schedule (default: 600:4e-4,1000:2e-4)
  --eps_sched           eps decay schedule (default: 100:0.1,3200:0.01)
  --reg                 reg
```
For example, if you run the following command:
```
python src/train_dpiqn.py --gpu=1 --mt --mt_mode=coop-only --eps_sched='100:0.1,3200:0.01' 
```
Then it will start training a DPIQN model in 2 vs. 2 soccer game, and it will only infer its coolaborator's policy. Besides, the eps parameter for epsilon-greedy will decrease to 0.1 at epoch 100, and down to 0.01 at epochj 3200. 

# Testing
To test the model, enter the command:
```
 python src/train_dpiqn.py --load=[path_to_model] --task=eval
```
The model will be evaluated for 100,000 episodes. In addition, you can use the following command to watch how your agent play:
```
 python src/train_dpiqn.py --load=[path_to_model] --task=play
```
Note that you can also use the same optional arguments listed in Training section.
