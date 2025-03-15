from distar.agent.default.replay_decoder import ReplayDecoder
from distar.ctools.utils import read_config, deep_merge_dicts
import torch
import torch.nn as nn
import torch.optim as optim
from distar.ctools.utils import read_config, deep_merge_dicts
from distar.agent.import_helper import import_module

# Updated import to use absolute paths rather than relative import.
from distar.agent.default.model.head.action_type_head import ActionTypeHead
from distar.agent.default.sl_training.sl_loss import SupervisedLoss
from distar.ctools.worker.learner.base_learner import BaseLearner
from distar.agent.default.model import Model
from distar.agent.default.sl_training.sl_dataloader import SLDataloader

import pickle, gzip, os

cfg = read_config('/Users/yeren/DI-star/distar/ctools/worker/learner/base_learner_default_config.yaml')
cfg = deep_merge_dicts(cfg, read_config('/Users/yeren/DI-star/distar/agent/default/model/actor_critic_default_config.yaml'))
cfg = deep_merge_dicts(cfg, read_config('/Users/yeren/DI-star/distar/bin/sl_user_config.yaml'))
cfg = deep_merge_dicts(cfg, read_config('/Users/yeren/DI-star/distar/bin/user_config.yaml'))

replay_decoder = ReplayDecoder(cfg)
data_path = os.path.abspath('.') + '/replay' #+ '_one/'
paths = [ data_path+f for f in os.listdir(data_path) if f.endswith('.SC2Replay') ] 
cfg.learner.data.train_data_file = data_path
cfg.learner.data.num_workers = 2

if __name__ == '__main__':

    if not os.path.exists('data.pkl'):
        dataloader = SLDataloader(cfg)
        data = next(dataloader)
        print(data)

        if data is not None:
            with open('data.pkl', 'wb') as f:
                pickle.dump(data, f)

    learning_rate = 1e-3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SLLearner = import_module(cfg.learner.agent, 'SLLearner')
    action_type_head = ActionTypeHead(cfg).to(device)
    loss_module = SupervisedLoss(cfg)  
    optimizer = optim.Adam(action_type_head.parameters(), lr=learning_rate)
    _model = Model(cfg, temperature=1.0)
    num_layers = cfg.model.encoder.core_lstm.num_layers
    hidden_size = cfg.model.encoder.core_lstm.hidden_size
    zero_tensor = torch.zeros(cfg.learner.data.batch_size, hidden_size)
    hidden_state = [(zero_tensor, zero_tensor) for _ in range(num_layers)]

    data = pickle.load(open('data.pkl', 'rb'))
    logits, infer_action_info, hidden_state = _model.sl_train(data, hidden_state=hidden_state)
    print(logits)