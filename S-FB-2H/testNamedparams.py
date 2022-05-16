import torch
from model import PCMLP
import os



model= PCMLP(5,5,0,0.01,0.33,0.33)
checkpointPhase = torch.load(os.path.join('models',f"PC_FF_E49_I0.pth"))
model.load_state_dict(checkpointPhase["module"])
print(model.named_parameters())
for name, p in model.named_parameters():
    print(name)