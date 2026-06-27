from pathlib import Path
import argparse
import torch

class Actor(torch.nn.Sequential):
    def __init__(self):
        super().__init__(torch.nn.Linear(50,512),torch.nn.ELU(),torch.nn.Linear(512,256),
                         torch.nn.ELU(),torch.nn.Linear(256,128),torch.nn.ELU(),
                         torch.nn.Linear(128,12))

if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("output",type=Path);a=p.parse_args()
    state=torch.load(a.checkpoint,map_location="cpu")["model_state_dict"]
    actor=Actor().eval();actor.load_state_dict({k[6:]:v for k,v in state.items() if k.startswith("actor.")})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(actor,torch.zeros(1,50),a.output,input_names=["observations"],output_names=["raw_actions"],
                      dynamic_axes={"observations":{0:"batch"},"raw_actions":{0:"batch"}},opset_version=17)
    print(f"Exported {a.output}")
