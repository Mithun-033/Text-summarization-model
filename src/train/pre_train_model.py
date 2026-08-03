import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch._inductor.config as inductor_config
from model.config import train_config, model_config
from model.model import SummarizationModel

# TODO : Async model checkpointing
# TODO : train & eval step 
# TODO : weight_decay scheduler
# TODO : model checkpoint loading
# TODO : docs and unittests


# Inductor config changes to make torch.compile() better.

inductor_config.fx_graph_cache = True
# Caches a part of compilation process, reduces cold satrt time for compilation
inductor_config.autograd_cache = True
# Extends the caching, saves more time on re-compilation
inductor_config.reorder_for_locality_in_training = True
# I couldn't understand this optimization, but it has something to do with better use of Hardware cache registries (L1/L2...)
# If anyone finds an explanation replace these lines with that.
inductor_config.max_autotune_prune_choices_based_on_shared_mem = True
# This is huge, basically doesn't trigger OOM at runtime, limits compilation to hardware limtis
inductor_config.memory_planning = True
# Apparently reduces peak_memory by 20-40% on static models, don't know the exact mechanism. 
def train_step(): ...


def eval_step(): ...


def train(
    model_path: str | None,
    train_conf: train_config,
    model_conf: model_config,
    epochs: int = 1,
    lr: float = 1e-4,
    batch_size: int = 1024,
    weight_decay: float = 0.01,
    compile: bool = False,
    device="cpu",
):

    if model_path :
        ...

    else:
        model = SummarizationModel(model_conf).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = ...

        warmup_scheduler = optim.lr_scheduler.LinearLR(...)
        # skipped lr decay as per modern methods, refer to docs shared for further info on why

        weight_decay_scheduler = ...

        if compile and device == "cuda":
            model = torch.compile(
            model,
            fullgraph=True,  # tries to compile the entire model in a single graph, seems like a no-brainer to do since our model is static
            options = {
                "epilogue_fusion" : True, # basically minimized kernel launches and tries to fuse many operations into a single kernel launch
                "max_autotune" : True, # uses triton to optimizer mat mults, again a no brainer for a static model like ours
                "shape_padding" : True, # basically GPU's perform better if tensors have shape of a power of 2, 
                # so a tensor with dim 1000 might be slower than a tensor with shape 1024, hence we do this padding.
                # Shld check for vram memory spike, disable if too high
                "triton.cudagraphs" : True # apparently reduces python overhead according to docs       
            }  
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument(
        "--compile", action="store_true"
    )  # pass the flag --compile in cli to enable compiling
    parser.add_argument("--load_checkpoint", type="str")

    args = parser.parse_args()

    train_conf = train_config(
        lr=args.lr, batch_size=args.batch_size, weight_decay=args.weight_decay
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        model_path=args.load_checkpoint,
        train_conf=train_conf,
        model_conf=model_config(),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        compile=args.compile,
        device=device,
    )
