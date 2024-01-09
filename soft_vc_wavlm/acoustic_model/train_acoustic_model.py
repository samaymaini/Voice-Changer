import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from acoustic_model import AcousticModel
from mel_dataset import MelDataset


BATCH_SIZE = 16
LEARNING_RATE = 4e-4
BETAS = (0.8, 0.99)
WEIGHT_DECAY = 1e-5
STEPS = 9000
LOG_INTERVAL = 5
VALIDATION_INTERVAL = 500
CHECKPOINT_INTERVAL = 500
BACKEND = "nccl"
INIT_METHOD = "tcp://localhost:54320"

def save_checkpoint(
    checkpoint_dir,
    acoustic,
    optimizer,
    step,
    loss,
    best,
):
    state = {
        "acoustic-model": acoustic.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }
    #checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model-{step}-{loss}.pt")
    torch.save(state, checkpoint_path)
    if best:
        best_path = os.path.join(checkpoint_dir, "model-best.pt")
        torch.save(state, best_path)
    print(f"Saved checkpoint: {os.path.basename(checkpoint_path)}")

def train(rank, world_size, args):
    dist.init_process_group(
        BACKEND,
        rank=rank,
        world_size=world_size,
        init_method=INIT_METHOD,
    )

    acoustic = AcousticModel(discrete=True).to(rank)

    acoustic = DDP(acoustic, device_ids=[rank])

    optimizer = optim.AdamW(
        acoustic.parameters(),
        lr=LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    if args["ckpt"] is not None:
        state_dict = torch.load(args["ckpt"])
        model_state_dict = state_dict["acoustic-model"]
        #reconfigured_model_state = {}
        #for key in model_state_dict.keys():
        #    reconfigured_model_state[key[7:]] = model_state_dict[key]
        acoustic.load_state_dict(model_state_dict)

        optim_state_dict = state_dict["optimizer"]
        optimizer.load_state_dict(optim_state_dict)


    train_dataset = MelDataset("../../../LJSpeech-1.1/", train=True, discrete=True)
    train_sampler = DistributedSampler(train_dataset, drop_last=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=train_dataset.pad_collate,
        num_workers=8,
        pin_memory=True,
        shuffle=False,
        drop_last=True
    )

    validation_dataset = MelDataset(
        root="../../../LJSpeech-1.1/",
        train=False
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    n_epochs = STEPS // len(train_loader) + 1
    global_step, best_loss = 0, float('inf')

    tepoch = tqdm(range(0, n_epochs + 1))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        train_sampler.set_epoch(epoch)

        acoustic.train()
        
        for mels, mels_lengths, units, units_lengths in train_loader:
            mels, mels_lengths = mels.to(rank), mels_lengths.to(rank)
            units, units_lengths = units.to(rank), units_lengths.to(rank)

            #print(mels.shape, units.shape)

            optimizer.zero_grad()

            mels_ = acoustic(units, mels)

            loss = F.l1_loss(mels_, mels, reduction="none")
            loss = torch.sum(loss, dim=(1, 2)) / (mels_.size(-1) * mels_lengths)
            loss = torch.mean(loss)
            if rank == 0:
                print(loss)

            loss.backward()
            optimizer.step()

            global_step += 1

            if global_step % VALIDATION_INTERVAL == 0:
                acoustic.eval()

                for i, (mels, units) in enumerate(validation_loader, 1):
                    mels, units = mels.to(rank), units.to(rank)

                    with torch.no_grad():
                        mels_ = acoustic(units, mels)
                        loss = F.l1_loss(mels_, mels)

                acoustic.train()

                new_best = best_loss > loss
                if new_best or global_step % CHECKPOINT_INTERVAL == 0:
                    if new_best:
                        print("-------- new best model found!")
                        best_loss = loss

                    if rank == 0:
                        save_checkpoint(
                            checkpoint_dir="ckpts",
                            acoustic=acoustic,
                            optimizer=optimizer,
                            step=global_step,
                            loss=loss,
                            best=new_best,
                        )

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    args = {
        "ckpt": None
    }

    mp.spawn(
        train,
        args=(world_size, args),
        nprocs=world_size,
        join=True
        )