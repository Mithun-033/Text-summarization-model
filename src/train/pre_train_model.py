import argparse
from typing import Any

import torch
import torch._inductor.config as inductor_config
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data.dataloaders import dataloader
from model.config import model_config, train_config
from model.model import SummarizationModel

# TODO : Async model checkpointing
# TODO : train & eval step (done)
# TODO : model checkpoint loading (done)
# TODO : docs and unittests
# TODO : proper logging of losses, find a alternative for .json 
# TODO : failsafe model saving

# Inductor config changes to make torch.compile() better.

inductor_config.fx_graph_cache = True
# Caches a part of compilation process, reduces cold satrt time for re-compilation
inductor_config.autograd_cache = True
# Extends the caching, saves more time on re-compilation
inductor_config.reorder_for_locality_in_training = True
# I couldn't understand this optimization, but it has something to do with better use of Hardware cache registries (L1/L2...)
# If anyone finds an explanation replace these lines with that.
inductor_config.max_autotune_prune_choices_based_on_shared_mem = True
# This is huge, basically doesn't trigger OOM at runtime, limits compilation to hardware limtis
inductor_config.memory_planning = True
# Apparently reduces peak_memory by 20-40% on static models, don't know the exact mechanism.


def train_step(
    model: SummarizationModel,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler,
    criterion: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    epoch: int,
    epochs: int,
    gradient_accumulation_steps: int = 1
):
    model.train()
    loss = 0.0
    temp_steps = 0
    with tqdm(train_loader, desc=f"{epoch + 1} / {epochs} :") as pbar:
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(x)
            loss += criterion(out.unsqueeze(1), y) / gradient_accumulation_steps
            temp_steps += 1

            if temp_steps % gradient_accumulation_steps == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix({"loss": loss.item()})
                loss = 0.0

    return temp_steps


def eval_step(
        model : SummarizationModel, 
        val_loader: torch.utils.data.DataLoader, 
        criterion: nn.Module, 
        epoch: int, 
        epochs: int
    ):
    model.eval()
    loss = 0.0
    with tqdm(val_loader, desc= f"Val {epoch + 1} / {epochs} :") as pbar:
        for x,y in pbar:
            x = x.to(device, non_blocking = True)
            y = y.to(device, non_blocking = True)

            out = model(x)
            loss += criterion(out.unsqueeze(1),y)
    return loss / len(val_loader)

def train(
    model_path: str | None,
    train_conf: train_config,
    model_conf: model_config,
    epochs: int = 1,
    compile: bool = False,
    device="cpu",
    debug_compile: bool = False
):

    model = SummarizationModel(model_conf).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamw(
        model.parameters(),
        lr=train_conf.lr,
        weight_decay=train_conf.weight_decay,
        fused=True,
    )  # just a placeholder, shld use Muon-Adamw
    scheduler = optim.lr_scheduler.LinearLR(optimizer, ...)

    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        model_state, optimizer_state, scheduler_state, steps = state_dict["model"], state_dict["optimizer"], state_dict["scheduler"], state_dict["steps"]
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)

    loader = dataloader(train_conf.batch_size, model_conf.max_seq_len, model_conf.vocab_size)
    train_dataloader, val_dataloader = (
        loader.train_dataloader(),
        loader.val_dataloader(),
    )
    if compile and device == "cuda":
        model = torch.compile(
            model,
            fullgraph=True,  # tries to compile the entire model in a single graph, seems like a no-brainer to do since our model is static
            options={
                "epilogue_fusion": True,  # basically minimized kernel launches and tries to fuse many operations into a single kernel launch
                "max_autotune": True,  # uses triton to optimizer mat mults, again a no brainer for a static model like ours
                "shape_padding": True,  # basically GPU's perform better if tensors have shape of a power of 2,
                # so a tensor with dim 1000 might be slower than a tensor with shape 1024, hence we do this padding.
                # Shld check for vram memory spike & disable if too high
                "triton.cudagraphs": True, # reduces python overhead according to docs
                "trace.enabled" : debug_compile,  # enables tracing  
                "trace.graph" : debug_compile  # shows fusion graph
            },
        )
    steps = 0
    gradient_accumulation_steps = train_conf.grad_batch_size // train_conf.batch_size  
    for epoch in range(epochs):
        ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--compile", action="store_true")  # pass the flag --compile in CLI to enable compiling
    parser.add_argument("--load_checkpoint", type="str")
    parser.add_argument("--debug_compile", action="store_true")

    args = parser.parse_args()

    train_conf = train_config(lr=args.lr, batch_size=args.batch_size, weight_decay=args.weight_decay)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        model_path=args.load_checkpoint,
        train_conf=train_conf,
        model_conf=model_config(),
        epochs=args.epochs,
        compile=args.compile,
        device=device, 
        debug_compile = args.debug_compile
    )
