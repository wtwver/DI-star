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

import pickle, gzip, os

batch_size = 32
input_dim = 128      
scalar_dim = 64       
num_classes = 10      
learning_rate = 1e-3
num_epochs = 50
data_path = os.path.abspath('.') + '/replay' + '_one/'
print(data_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    cfg = read_config('/Users/yeren/DI-star/distar/agent/default/model/actor_critic_default_config.yaml')
    cfg = deep_merge_dicts(cfg, read_config('/Users/yeren/DI-star/distar/bin/sl_user_config.yaml'))
    cfg = deep_merge_dicts(cfg, read_config('/Users/yeren/DI-star/distar/bin/user_config.yaml'))
    cfg.learner.data.train_data_file = data_path

    SLLearner = import_module(cfg.learner.agent, 'SLLearner')
    action_type_head = ActionTypeHead(cfg).to(device)
    loss_module = SupervisedLoss(cfg)  
    optimizer = optim.Adam(action_type_head.parameters(), lr=learning_rate)
    _model = Model(cfg, temperature=1.0)
    num_layers = cfg.model.encoder.core_lstm.num_layers
    hidden_size = cfg.model.encoder.core_lstm.hidden_size
    zero_tensor = torch.zeros(cfg.learner.data.batch_size, hidden_size)

    sllearn = SLLearner(cfg)


    exit()
    hidden_state = [(zero_tensor, zero_tensor) for _ in range(num_layers)]

    logits, infer_action_info, hidden_state = _model.sl_train(**data, hidden_state=hidden_state)

    for epoch in range(num_epochs):
        lstm_output = torch.randn(batch_size, input_dim).to(device)
        scalar_context = torch.randn(batch_size, scalar_dim).to(device)
        ground_truth = torch.randint(0, num_classes, (batch_size,)).to(device)
        mask = torch.ones(batch_size, device=device)
        
        logits, _, _ = action_type_head(lstm_output, scalar_context, ground_truth)
        
        loss = loss_module.action_type_loss(logits, ground_truth, mask)
        
        if epoch > 5:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")