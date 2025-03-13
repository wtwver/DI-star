import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import distar.pysc2

# --- Placeholder: Replay Parsing using pysc2 ---
#
# Replace the logic below with the actual pysc2 function calls you use in your codebase 
# to load and parse an SC2 replay. The goal is to return a list of samples, where each sample
# is a tuple (features, context, action).
#
def parse_replay(replay_path, input_dim, context_dim):
    """
    Parse an SC2 replay file using pysc2 and extract training examples.
    Each sample should be a tuple:
      (features, context, action)
    where:
      - features: a list of floats of length=input_dim (e.g. game state feature vector)
      - context: a list of floats of length=context_dim (e.g. additional context data)
      - action: an integer representing the ground-truth action type
      
    NOTE: This is a placeholder. Replace this dummy implementation with your real replay-parsing logic.
    """
    samples = []
    try:
        # Example pseudocode: Replace with your actual pysc2 replay parsing code.
        from distar.pysc2.lib import replay as sc2_replay  # Adjust the import as needed
        replay_data = sc2_replay.load_replay(replay_path)  # This is pseudocode
        
        # Iterate over events in the replay and extract training samples.
        for event in replay_data.events:
            # Replace the event filtering and extraction with your logic.
            if event.type == "player_action":  # dummy filter condition
                # Example extraction; adjust field names as per your replay object.
                features = event.state_features if hasattr(event, 'state_features') else [0.0] * input_dim
                context = event.context_vector if hasattr(event, 'context_vector') else [0.0] * context_dim
                action = event.action_type if hasattr(event, 'action_type') else 0
                if len(features) == input_dim and len(context) == context_dim:
                    samples.append((features, context, action))
    except Exception as e:
        print(f"Failed to parse replay {replay_path}: {e}")
    return samples

# --- Dataset Definition ---
#
# SC2ReplayDataset reads all replays from a given directory (files ending in .SC2Replay)
# and extracts training examples using the parse_replay function.
#
class SC2ReplayDataset(Dataset):
    def __init__(self, replay_dir, input_dim, context_dim):
        self.samples = []
        # Find all replay files in the directory.
        replay_files = glob.glob(os.path.join(replay_dir, "*.SC2Replay"))
        for replay_file in replay_files:
            replay_samples = parse_replay(replay_file, input_dim, context_dim)
            self.samples.extend(replay_samples)
        if not self.samples:
            raise ValueError("No training samples found—check your replay parsing logic.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        features, context, action = self.samples[idx]
        features = torch.tensor(features, dtype=torch.float32)
        context = torch.tensor(context, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        return features, context, action

# --- Helper Functions and Model Components ---
#
def fc_block(in_dim, out_dim, activation=None):
    layers = [nn.Linear(in_dim, out_dim)]
    if activation is not None:
        layers.append(activation)
    return nn.Sequential(*layers)

class ResFCBlock(nn.Module):
    def __init__(self, dim, activation=nn.ReLU()):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.act = activation
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)
        return self.act(out + residual)

def build_activation(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "glu":
        # A simple GLU-like block that takes two inputs and returns a gated output.
        class SimpleGLU(nn.Module):
            def __init__(self, in_dim, out_dim, context_dim):
                super().__init__()
                self.fc = nn.Linear(in_dim, out_dim)
                self.context_fc = nn.Linear(context_dim, out_dim)
            def forward(self, x, context):
                out = self.fc(x) + self.context_fc(context)
                return out * torch.sigmoid(out)
        return lambda in_dim, out_dim, context_dim: SimpleGLU(in_dim, out_dim, context_dim)
    else:
        return nn.ReLU()

# --- Model Definition: ActionTypeHead ---
#
# This model definition is inspired by your codebase; it creates a projection layer,
# a residual stack, and uses a GLU-like block to compute action logits.
#
class ActionTypeHead(nn.Module):
    def __init__(self, cfg):
        super(ActionTypeHead, self).__init__()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg["model"]["policy"]["head"]["action_type_head"]
        self.act = build_activation(self.cfg["activation"])
        self.project = fc_block(self.cfg["input_dim"], self.cfg["res_dim"], activation=self.act)
        blocks = [ResFCBlock(self.cfg["res_dim"], activation=self.act) for _ in range(self.cfg["res_num"])]
        self.res = nn.Sequential(*blocks)
        self.drop_ratio = self.cfg.get("drop_ratio", 0.0)
        self.drop_Z = nn.Dropout(p=self.drop_ratio)
        self.action_fc = build_activation("glu")(self.cfg["res_dim"], self.cfg["action_num"], self.cfg["context_dim"])
        self.action_map_fc1 = fc_block(self.cfg["action_num"], self.cfg["action_map_dim"], activation=self.act)
        self.action_map_fc2 = fc_block(self.cfg["action_map_dim"], self.cfg["action_map_dim"])
        self.glu1 = build_activation("glu")(self.cfg["action_map_dim"], self.cfg["gate_dim"], self.cfg["context_dim"])
        self.glu2 = build_activation("glu")(self.cfg["input_dim"], self.cfg["gate_dim"], self.cfg["context_dim"])
        self.action_num = self.cfg["action_num"]
        if self.whole_cfg["common"]["type"] == "play":
            self.use_mask = True
        else:
            self.use_mask = False
        self.race = "zerg"  # Placeholder for masking logic

    def forward(self, lstm_output, scalar_context, action_type=None):
        x = self.project(lstm_output)
        x = self.res(x)
        x = self.action_fc(x, scalar_context)
        x = x / self.whole_cfg["model"]["temperature"]
        if action_type is None:
            p = F.softmax(x, dim=1)
            action_type = torch.multinomial(p, num_samples=1)[:, 0]
        action_one_hot = F.one_hot(action_type.long(), num_classes=self.action_num).float()
        embedding1 = self.action_map_fc1(action_one_hot)
        embedding1 = self.action_map_fc2(embedding1)
        embedding1 = self.glu1(embedding1, scalar_context)
        embedding2 = self.glu2(lstm_output, scalar_context)
        embedding = embedding1 + embedding2
        return x, action_type, embedding

# --- Training Function ---
#
def train(model, dataloader, optimizer, epochs, device):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for features, context, gt_actions in dataloader:
            features = features.to(device)
            context = context.to(device)
            gt_actions = gt_actions.to(device)
            optimizer.zero_grad()
            logits, pred_actions, embedding = model(features, context, gt_actions)
            loss = F.cross_entropy(logits, gt_actions)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * features.size(0)
        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}")

# --- Main: Argument parsing and launching the training ---
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ActionTypeHead using SC2 replays via pysc2")
    parser.add_argument("--replay_dir", type=str, required=True, help="Directory containing SC2 replay files")
    parser.add_argument("--epochs", type=int, default=100, help="No. of epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    args = parser.parse_args()

    # Configuration mimicking your model settings; adjust appropriately
    config = {
        "model": {
            "policy": {
                "head": {
                    "action_type_head": {
                        "activation": "relu",
                        "input_dim": 128,      # Feature dimension extracted from replay state
                        "res_dim": 64,
                        "res_num": 2,
                        "drop_ratio": 0.1,
                        "action_num": 10,      # Total number of action types
                        "action_map_dim": 32,
                        "gate_dim": 16,
                        "context_dim": 8       # Dimension of additional context extracted from replay
                    }
                }
            },
            "temperature": 1.0,
        },
        "common": {
            "type": "train"
        }
    }

    # Initialize the model.
    model = ActionTypeHead(config)
    # Create the SC2 replay dataset.
    dataset = SC2ReplayDataset(args.replay_dir,
                               input_dim=config["model"]["policy"]["head"]["action_type_head"]["input_dim"],
                               context_dim=config["model"]["policy"]["head"]["action_type_head"]["context_dim"])
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Begin training.
    train(model, dataloader, optimizer, args.epochs, args.device)