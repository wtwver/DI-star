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
from distar.agent.default.sl_training.sl_loss import SupervisedLoss

import pickle, gzip, os
from pprint import pprint

cfg = read_config('/Users/yeren/DI-star/distar/ctools/worker/learner/base_learner_default_config.yaml')
cfg = deep_merge_dicts(cfg, read_config('/Users/yeren/DI-star/distar/agent/default/model/actor_critic_default_config.yaml'))
cfg = deep_merge_dicts(cfg, read_config('/Users/yeren/DI-star/distar/bin/sl_user_config.yaml'))
cfg = deep_merge_dicts(cfg, read_config('/Users/yeren/DI-star/distar/bin/user_config.yaml'))

replay_decoder = ReplayDecoder(cfg)
data_path = os.path.abspath('.') + '/replay' #+ '_one/'
paths = [ data_path+f for f in os.listdir(data_path) if f.endswith('.SC2Replay') ] 
cfg.learner.data.train_data_file = data_path
cfg.learner.data.num_workers = 2

import os.path as osp
alphastar_model_default_config = read_config('/Users/yeren/DI-star/distar/agent/default/model/actor_critic_default_config.yaml')
from distar.agent.default.model.encoder import Encoder
from distar.agent.default.model.policy import Policy
from distar.agent.default.model.obs_encoder.value_encoder import ValueEncoder
from distar.agent.default.model.lstm import script_lnlstm
from torch import Tensor
from typing import Dict, Tuple, List
class AlphaStar(nn.Module):
    def __init__(self, cfg={}, use_value_network=False, temperature=None):
        super(AlphaStar, self).__init__()
        self.whole_cfg = deep_merge_dicts(alphastar_model_default_config, cfg)
        if temperature is not None:
            self.whole_cfg.model.temperature = temperature
        self.cfg = self.whole_cfg.model
        self.encoder = Encoder(self.whole_cfg)
        self.policy = Policy(self.whole_cfg)
        self._use_value_feature = self.whole_cfg.learner.get('use_value_feature',False)
        self.only_update_baseline = self.cfg.get('only_update_baseline', False)
        self.core_lstm = script_lnlstm(self.cfg.encoder.core_lstm.input_size,
                                       self.cfg.encoder.core_lstm.hidden_size,
                                       self.cfg.encoder.core_lstm.num_layers)
   
    def forward(self, spatial_info: Tensor, entity_info: Dict[str, Tensor], scalar_info: Dict[str, Tensor],
            entity_num: Tensor, hidden_state: List[Tuple[Tensor, Tensor]],
                ):
        lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip = \
            self.encoder(spatial_info, entity_info, scalar_info, entity_num)
        lstm_output, out_state = self.core_lstm(lstm_input.unsqueeze(dim=0), hidden_state)
        action_info, selected_units_num, logit, extra_units = self.policy(lstm_output.squeeze(dim=0), entity_embeddings, map_skip,
                                                             scalar_context, entity_num)
        return action_info, selected_units_num, out_state

    def sl_train(self,
                 spatial_info,
                 entity_info,
                 scalar_info,
                 entity_num,
                 selected_units_num,
                 traj_lens,
                 hidden_state,
                 action_info,
                 **kwargs):
        batch_size = len(traj_lens)
        lstm_input, scalar_context, baseline_feature, entity_embeddings, map_skip = \
            self.encoder(spatial_info, entity_info, scalar_info, entity_num)
        lstm_input = lstm_input.view(-1, lstm_input.shape[0] // batch_size, lstm_input.shape[-1]).permute(1, 0, 2)
        lstm_output, out_state = self.core_lstm(lstm_input, hidden_state)
        lstm_output = lstm_output.permute(1, 0, 2).contiguous().view(-1, lstm_output.shape[-1])
        action_info, selected_units_num, logits = self.policy.train_forward(lstm_output, entity_embeddings,
                                                                            map_skip, scalar_context, entity_num,
                                                                            action_info, selected_units_num)
        return logits, action_info, out_state


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
    # _model = Model(cfg, temperature=1.0)
    _model = AlphaStar(cfg, temperature=1.0)
    num_layers = cfg.model.encoder.core_lstm.num_layers
    hidden_size = cfg.model.encoder.core_lstm.hidden_size
    zero_tensor = torch.zeros(cfg.learner.data.batch_size, hidden_size)
    hidden_state = [(zero_tensor, zero_tensor) for _ in range(num_layers)]

    data = pickle.load(open('data.pkl', 'rb'))
    logits, infer_action_info, hidden_state = _model.sl_train(**data, hidden_state=hidden_state)
    # print(logits)

    _loss = SupervisedLoss(cfg)
    log_vars = _loss.compute_loss(logits, data['action_info'], data['action_mask'],
                                               data['selected_units_num'], data['entity_num'], infer_action_info)
    pprint(log_vars)