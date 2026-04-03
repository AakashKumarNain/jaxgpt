import os

import dataclasses
import warnings
import logging
import time
from pathlib import Path
from functools import partial

import jax
# Initialize distributed JAX (no-op for single process)
jax.distributed.initialize()

import optax
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint as ocp
from jax.experimental.multihost_utils import process_allgather
from jax.sharding import Mesh
from jax.sharding import set_mesh


from model import count_params
from model import precompute_frequencies
from model import GPT, forward
from utils import logical_to_sharding
from optim import build_optimizer
from config import ShardingRules, Config, BATCH_AXIS_NAME
from fineweb_dataloader import make_grain_shard_loader, BOSFinder


logging.getLogger("absl").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CheckpointManager.*")


def compute_loss(params, x_batch, y_batch, segment_ids, freqs, loss_mask):
    logits = forward(params, x_batch, segment_ids, freqs)
    if loss_mask is not None:
        per_token_loss = optax.losses.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=y_batch,
            where=loss_mask,
        )
        return jnp.sum(per_token_loss) / jnp.maximum(jnp.sum(loss_mask), 1.0)
    else:
        return jnp.mean(
            optax.losses.softmax_cross_entropy_with_integer_labels(
                logits=logits, labels=y_batch
            )
        )


@partial(
    jax.jit,
    static_argnames=("optim", "grad_accum_steps"),
    donate_argnames=("params", "x_batch", "y_batch", "optim_state"),
)
def train_step_accum(
    params,
    x_batch,
    y_batch,
    segment_ids,
    freqs,
    loss_mask,
    optim_state,
    optim,
    grad_accum_steps,
):
    def body(carry, xy):
        param, opt_state, lsum = carry
        xb, yb = xy
        loss, grad = jax.value_and_grad(compute_loss)(
            param, xb, yb, segment_ids, freqs, loss_mask
        )

        # MultiSteps accumulates grad internally and returns a zero-tree update on
        # every micro-step except the last, where it emits the real update.
        updates, new_opt_state = optim.update(grad, opt_state, param)
        new_param = optax.apply_updates(param, updates)
        return (new_param, new_opt_state, lsum + loss), None

    carry0 = (params, optim_state, jnp.array(0.0, dtype=jnp.result_type(0.0)))
    (params, optim_state, lsum), _ = jax.lax.scan(
        body, carry0, (x_batch, y_batch), length=grad_accum_steps
    )
    loss = lsum / grad_accum_steps
    return params, loss, optim_state


@partial(
    jax.jit,
    static_argnames=("optim",),
    donate_argnames=("params", "x_batch", "y_batch", "optim_state"),
)
def train_step(params, x_batch, y_batch, segment_ids, freqs, loss_mask, optim_state, optim):
    loss, grads = jax.value_and_grad(compute_loss)(
        params, x_batch, y_batch, segment_ids, freqs, loss_mask
    )
    updates, optim_state = optim.update(grads, optim_state, params)
    updated_params = optax.apply_updates(params, updates)
    return updated_params, loss, optim_state


@jax.jit
def val_step(params, x_batch, y_batch, segment_ids, freqs, loss_mask):
    loss = compute_loss(params, x_batch, y_batch, segment_ids, freqs, loss_mask)
    return loss


def line(label, value, comma=False, label_w=30, colon_w=2, value_w=20):
    fmt = f">{value_w}," if comma else f">{value_w}"
    return f"{label:<{label_w}}{':':<{colon_w}}{value:{fmt}}"


def get_next_batch(
    starts,
    ends,
    bsz,
    seqlen,
    tokens,
    data_sharding,
    buf_u16,
    transfer_to_device=False,
    create_new_buf=False,
):
    """Gathers batches of input-labels pairs.

    Given the `starts` and `ends` of sequences provided by the
    BOSFinder, this method generates batches of inputs-labels
    efficiently.
    """
    if buf_u16 is None and create_new_buf:
        buf_u16 = np.empty((bsz, seqlen + 1), dtype=np.uint16)

    ptr = 0
    for i, j in zip(starts, ends):
        n = j - i
        row = ptr // (seqlen + 1)
        col = ptr % (seqlen + 1)
        buf_u16[row, col : col + n] = tokens[i:j]
        ptr += n

    # If no new array was created
    if not create_new_buf:
        return
    else:
        if transfer_to_device:
            x = jax.device_put(buf_u16[:, :-1], data_sharding)
            y = jax.device_put(buf_u16[:, 1:], data_sharding)
        else:
            x = buf_u16[:, :-1]
            y = buf_u16[:, 1:]
        return x, y

def model_run_name(cfg):
    return (
        f"_L{cfg.model.num_layers}"
        f"_D{cfg.model.d_emb}"
        f"_Q{cfg.model.q_heads}"
        f"_KV{cfg.model.kv_heads}"
        f"_H{cfg.model.attn.head_dim}"
        f"_T{cfg.model.seqlen}"
        f"_V{cfg.model.vocab_size}"
        f"_{cfg.model.window_pattern}"
    )

@dataclasses.dataclass
class ShardCursor:
    tokens: object
    bos_finder: BOSFinder
    shard_name: str
    num_batches_in_shard: int
    file_index: int


def load_next_indexed_shard(data_iter, bsz, seqlen, next_file_index):
    shard = next(data_iter, None)
    if shard is None:
        return None, next_file_index

    tokens = shard["tokens"]
    bos_finder = BOSFinder(tokens)
    bos_finder.bos_idx = shard["bos_idx"]
    bos_finder.size = shard["size"]
    num_batches_in_shard = bos_finder.build(bsz, seqlen)
    shard_name = Path(shard["path"]).name

    return ShardCursor(
        tokens=tokens,
        bos_finder=bos_finder,
        shard_name=shard_name,
        num_batches_in_shard=num_batches_in_shard,
        file_index=next_file_index,
    ), next_file_index + 1


def release_shard_tokens(shard_cursor):
    if shard_cursor is not None:
        shard_cursor.tokens.unlink_on_del()


def build_train_progress_state(
    next_train_file_index,
    train_shard_cursor,
    num_shards_used,
    total_tokens_consumed,
    best_loss,
    last_val_loss,
    es_patience_counter,
    best_step,
):
    active_train_file_index = -1
    active_train_batch_iter = 0

    if train_shard_cursor is not None:
        active_train_file_index = train_shard_cursor.file_index
        active_train_batch_iter = train_shard_cursor.bos_finder.batch_iter

    return {
        "next_train_file_index": np.int64(next_train_file_index),
        "active_train_file_index": np.int64(active_train_file_index),
        "active_train_batch_iter": np.int64(active_train_batch_iter),
        "num_shards_used": np.int64(num_shards_used),
        "total_tokens_consumed": np.int64(total_tokens_consumed),
        "best_loss": np.float64(best_loss),
        "last_val_loss": np.float64(last_val_loss),
        "es_patience_counter": np.int64(es_patience_counter),
        "best_step": np.int64(best_step),
    }


def discard_loaded_shards(data_iter, shard_count):
    for discard_index in range(shard_count):
        shard = next(data_iter, None)
        if shard is None:
            break
        shard["tokens"].unlink_on_del()


def restore_train_shard_cursor(data_iter, train_progress_state, bsz, seqlen):
    next_train_file_index = int(train_progress_state["next_train_file_index"])
    active_train_file_index = int(train_progress_state["active_train_file_index"])
    active_train_batch_iter = int(train_progress_state["active_train_batch_iter"])

    if active_train_file_index < 0:
        discard_loaded_shards(data_iter, next_train_file_index)
        return None, next_train_file_index

    discard_loaded_shards(data_iter, active_train_file_index)
    train_shard_cursor, next_train_file_index = load_next_indexed_shard(
        data_iter, bsz, seqlen, active_train_file_index
    )
    if train_shard_cursor is None:
        return None, next_train_file_index

    active_train_batch_iter = min(
        active_train_batch_iter, train_shard_cursor.num_batches_in_shard
    )
    train_shard_cursor.bos_finder.batch_iter = active_train_batch_iter
    train_shard_cursor.bos_finder.i = int(
        train_shard_cursor.bos_finder.built_ptrs[active_train_batch_iter]
    )
    return train_shard_cursor, next_train_file_index


def prepare_local_batch_buffer(
    data_iter,
    shard_cursor,
    bsz,
    seqlen,
    data_sharding,
    batch_buffer,
    next_file_index,
):
    loaded_shards = []
    completed_shards = 0

    while True:
        if shard_cursor is None:
            shard_cursor, next_file_index = load_next_indexed_shard(
                data_iter, bsz, seqlen, next_file_index
            )
            if shard_cursor is None:
                return (
                    False,
                    shard_cursor,
                    completed_shards,
                    loaded_shards,
                    next_file_index,
                )

            loaded_shards.append(
                (shard_cursor.shard_name, shard_cursor.num_batches_in_shard)
            )
            if shard_cursor.num_batches_in_shard <= 0:
                release_shard_tokens(shard_cursor)
                shard_cursor = None
                completed_shards += 1
                continue

        try:
            starts, ends = shard_cursor.bos_finder.next_batch(bsz, seqlen)
            get_next_batch(
                starts,
                ends,
                bsz,
                seqlen,
                shard_cursor.tokens,
                data_sharding,
                batch_buffer,
                transfer_to_device=False,
            )
            return (
                True,
                shard_cursor,
                completed_shards,
                loaded_shards,
                next_file_index,
            )
        except StopIteration:
            release_shard_tokens(shard_cursor)
            shard_cursor = None
            completed_shards += 1


def main():

    process_index = jax.process_index()
    num_processes = jax.process_count()

    # Get the mesh, sharding rules, amd the config
    devices = np.array(jax.devices())
    if process_index == 0:
        print("Number of devices found:", len(devices))
        print("Number of processes:", num_processes)
    mesh = Mesh(devices, axis_names=BATCH_AXIS_NAME)
    sharding_rules = ShardingRules(batch=BATCH_AXIS_NAME)
    cfg = Config(mesh=mesh, rules=sharding_rules)

    train_files = sorted(Path(cfg.data_dir).glob("*train*.bin"))
    val_files = sorted(Path(cfg.data_dir).glob("*val*.bin"))

    # Shard training files across processes (round-robin)
    train_files = train_files[process_index::num_processes]
    val_files = val_files[process_index::num_processes]

    num_train_files = len(train_files)
    num_val_files = len(val_files)
    if process_index == 0:
        print("\nNumber of train files found (this process): ", num_train_files)
        print("Number of validation files found (this process): ", num_val_files)

    train_dl = make_grain_shard_loader(train_files)
    val_dl = make_grain_shard_loader(val_files, max_workers=1, max_buffer_size=256)
    train_iter = iter(train_dl)

    per_device_bsz = cfg.hparams.per_device_batch_size
    local_device_count = jax.local_device_count()
    bsz = per_device_bsz * local_device_count
    seqlen = cfg.model.seqlen
    head_dim = cfg.model.attn.head_dim
    data_sharding = logical_to_sharding(("batch",), cfg.mesh, cfg.rules)
    data_accum_sharding = logical_to_sharding(
        (None, "batch", None), cfg.mesh, cfg.rules
    )

    max_lr = cfg.hparams.max_lr
    min_lr = cfg.hparams.min_lr
    warmup_steps = cfg.hparams.warmup_steps
    desired_batch_size = cfg.hparams.desired_batch_size
    global_bsz = per_device_bsz * jax.device_count()
    global_tokens_per_microbatch = global_bsz * seqlen
    grad_accum_steps = max(1, (desired_batch_size + global_tokens_per_microbatch - 1) // global_tokens_per_microbatch)  # fmt: off
    total_train_steps = cfg.hparams.total_train_steps
    max_checkpoints_to_keep = cfg.ckpt_cfg.max_checkpoints_to_keep
    checkpoint_save_steps = cfg.ckpt_cfg.checkpoint_save_steps

    # Load the model
    if process_index == 0:
        print("Building GPT model based on the config...")
    model = GPT.init(jax.random.PRNGKey(0), cfg)
    if process_index == 0:
        print("Model built successfully!")

    # Optimizer
    optim = optax.chain(
        optax.clip_by_global_norm(cfg.hparams.grad_clip_norm),
        build_optimizer(
            model,
            d_model=cfg.model.d_emb,
            other_peak_lr=max_lr,
            other_min_lr=min_lr,
            total_train_steps=total_train_steps,
            warmup_steps=warmup_steps,
            b1=cfg.hparams.b1,
            b2=cfg.hparams.b2,
            embedding_lr=cfg.hparams.embedding_lr,
            weight_decay=cfg.hparams.weight_decay,
            cautious_weight_decay=cfg.hparams.cautious_weight_decay,
        ),
    )

    if grad_accum_steps > 1:
        if process_index == 0:
            print("Using `MultiSteps` in optax for gradient accumulation...")
        optim = optax.MultiSteps(optim, every_k_schedule=grad_accum_steps)

    optim_state = optim.init(model)

    # Checkpointing
    ckpt_path = Path(cfg.ckpt_cfg.save_ckpt_dir)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    
    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints_to_keep,
        save_interval_steps=checkpoint_save_steps,
        enable_async_checkpointing=True,
        enable_background_delete=True,
        single_host_load_and_broadcast=True
    )
    handlers = {
        "params": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "optim_state": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        "train_progress": ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
    }

    mngr = ocp.CheckpointManager(ckpt_path, handlers, options=options)

    if process_index == 0:
        print("")
        print("-" * 75)
        print("")
        print(line("Run name", model_run_name(cfg), value_w=30))
        print(line("Attention Pattern", cfg.model.window_pattern))
        print(line("Model dtype", str(cfg.model.dtype)))
        print(line("Num layers", cfg.model.num_layers))
        print(line("Embedding dim", cfg.model.d_emb))
        print(line("Query heads", cfg.model.q_heads))
        print(line("KV heads", cfg.model.kv_heads))
        print(line("Head dim", cfg.model.attn.head_dim))
        print(line("MLP hidden dim", cfg.model.mlp.fc1.out_features))
        print(line("Vocab size", cfg.model.vocab_size))
        print(line("Number of trainable params: ", count_params(model), comma=True))
        print(line("Sequence length per sample", seqlen))
        print(line("Per device batch size", per_device_bsz))
        print(line("Total batch size", bsz))
        print(line("Grad accumulation steps", grad_accum_steps))
        print()
        print(line("LR (min, max)", str((f"{min_lr:.6f}", f"{max_lr:.6f}"))))
        print(line("Warmup steps", cfg.hparams.warmup_steps))
        print(line("Weight decay", cfg.hparams.weight_decay), "\n")
        print("-" * 75)

    # Compute the frequencies
    positions = jnp.arange(seqlen)[None, :]
    with set_mesh(cfg.mesh):
        freqs = precompute_frequencies(positions=positions, features=head_dim)

    # Because our dataloader already ensures that sequence in a batch have
    # tokens equal to the context window, we do not need sequence packing here
    # Hence, we can segment_ids to None for pretraining.
    segment_ids = None
    resume_from_step = cfg.ckpt_cfg.last_checkpoint_step
    train_progress_state = build_train_progress_state(
        next_train_file_index=0,
        train_shard_cursor=None,
        num_shards_used=0,
        total_tokens_consumed=0,
        best_loss=float("inf"),
        last_val_loss=float("inf"),
        es_patience_counter=0,
        best_step=0,
    )
    next_train_file_index = 0
    train_shard_cursor = None

    if resume_from_step > 0:
        resume_ckpt_path = os.path.join(
            cfg.ckpt_cfg.save_ckpt_dir, str(resume_from_step)
        )
        if os.path.exists(resume_ckpt_path):
            restored_checkpoint = mngr.restore(
                resume_from_step,
                args=ocp.args.Composite(
                    params=ocp.args.PyTreeRestore(item=model),
                    optim_state=ocp.args.PyTreeRestore(item=optim_state),
                    train_progress=ocp.args.PyTreeRestore(item=train_progress_state),
                ),
            )
            model = restored_checkpoint.params
            optim_state = restored_checkpoint.optim_state
            train_progress_state = restored_checkpoint.train_progress
            train_shard_cursor, next_train_file_index = restore_train_shard_cursor(
                train_iter, train_progress_state, bsz, seqlen
            )
        else:
            resume_from_step = 0
            if process_index == 0:
                print(
                    f"Checkpoint path {resume_ckpt_path} not found! Resuming training without restoring checkpoint..."
                )

    best_loss = float(train_progress_state["best_loss"])
    last_val_loss = float(train_progress_state["last_val_loss"])
    es_patience = cfg.hparams.es_patience
    es_patience_counter = int(train_progress_state["es_patience_counter"])
    best_step = int(train_progress_state["best_step"])
    num_shards_used = int(train_progress_state["num_shards_used"])
    total_tokens_consumed = int(train_progress_state["total_tokens_consumed"])

    simple_batch = np.zeros((bsz, seqlen + 1), dtype=np.uint16)
    grad_accum_batch = np.zeros((grad_accum_steps, bsz, seqlen + 1), dtype=np.uint16)
    val_data_buf = np.zeros((bsz, seqlen + 1), dtype=np.uint16)
    microbatches_per_step = grad_accum_steps if grad_accum_steps > 1 else 1
    val_interval = max(1, cfg.hparams.val_interval)

    step = resume_from_step
    if process_index == 0:
        print("Starting training (the first step will take some time for compilation...)\n")

    training_complete = False
    stopped_for_data_exhaustion = False
    train_start_time = time.time()

    while not training_complete and step < total_train_steps:
        loaded_train_shards = []
        completed_train_shards = 0
        local_train_step_ready = True

        if grad_accum_steps > 1:
            for micro_step in range(grad_accum_steps):
                (
                    batch_ready,
                    train_shard_cursor,
                    closed_shards,
                    shard_logs,
                    next_train_file_index,
                ) = (
                    prepare_local_batch_buffer(
                        train_iter,
                        train_shard_cursor,
                        bsz,
                        seqlen,
                        data_accum_sharding,
                        grad_accum_batch[micro_step],
                        next_train_file_index,
                    )
                )
                completed_train_shards += closed_shards
                loaded_train_shards.extend(shard_logs)
                if not batch_ready:
                    local_train_step_ready = False
                    break
        else:
            (
                batch_ready,
                train_shard_cursor,
                closed_shards,
                shard_logs,
                next_train_file_index,
            ) = (
                prepare_local_batch_buffer(
                    train_iter,
                    train_shard_cursor,
                    bsz,
                    seqlen,
                    data_sharding,
                    simple_batch,
                    next_train_file_index,
                )
            )
            completed_train_shards += closed_shards
            loaded_train_shards.extend(shard_logs)
            local_train_step_ready = batch_ready

        num_shards_used += completed_train_shards

        if process_index == 0:
            for shard_name, shard_batch_count in loaded_train_shards:
                print(
                    f"\n=== Loaded Train Shard: {shard_name} | "
                    f"Indexed {shard_batch_count} batches ==="
                )

        local_train_step_flag = np.array(
            1 if local_train_step_ready else 0, dtype=np.int32
        )
        all_train_step_flags = np.asarray(
            process_allgather(local_train_step_flag, tiled=False)
        )
        if int(np.min(all_train_step_flags)) == 0:
            stopped_for_data_exhaustion = True
            if process_index == 0:
                print(
                    f"Stopping training because not all hosts can assemble "
                    f"another full train step at step {step}/{total_train_steps}."
                )
            break

        start = time.time()
        if grad_accum_steps > 1:
            stacked_batch = jnp.asarray(
                grad_accum_batch,
                dtype=jnp.int32,
                device=data_accum_sharding,
            )
            stacked_x = stacked_batch[:, :, :-1]
            stacked_y = stacked_batch[:, :, 1:]
            model, loss, optim_state = train_step_accum(
                model,
                stacked_x,
                stacked_y,
                segment_ids,
                freqs,
                None,
                optim_state,
                optim,
                grad_accum_steps,
            )
        else:
            stacked_batch = jnp.asarray(simple_batch, dtype=jnp.int32, device=data_sharding)
            stacked_x = stacked_batch[:, :-1]
            stacked_y = stacked_batch[:, 1:]
            model, loss, optim_state = train_step(
                model,
                stacked_x,
                stacked_y,
                segment_ids,
                freqs,
                None,
                optim_state,
                optim,
            )

        jax.block_until_ready(loss)
        end = time.time()
        dt = end - start
        train_time_elapsed = (end - train_start_time) / 60
        tokens_processed = global_bsz * seqlen * microbatches_per_step
        total_tokens_consumed += tokens_processed
        tokens_per_sec = int(tokens_processed / max(dt, 1e-6))

        if process_index == 0:
            print(
                f"Step: [{str(step).zfill(len(str(total_train_steps)))}/{total_train_steps}] | "
                f"loss: {loss:8.4f} | Step time: {dt:5.2f} s | "
                f"Train time: {train_time_elapsed:6.2f} min | "
                f"Tokens processed/s: {tokens_per_sec:>9,}"
            )

        step += 1

        if (step % options.save_interval_steps) == 0:
            train_progress_state = build_train_progress_state(
                next_train_file_index=next_train_file_index,
                train_shard_cursor=train_shard_cursor,
                num_shards_used=num_shards_used,
                total_tokens_consumed=total_tokens_consumed,
                best_loss=best_loss,
                last_val_loss=last_val_loss,
                es_patience_counter=es_patience_counter,
                best_step=best_step,
            )
            mngr.save(
                step,
                args=ocp.args.Composite(
                    params=ocp.args.PyTreeSave(model),
                    optim_state=ocp.args.PyTreeSave(optim_state),
                    train_progress=ocp.args.PyTreeSave(train_progress_state),
                ),
            )

        should_run_validation = (step % val_interval) == 0 or step >= total_train_steps
        if not should_run_validation:
            continue

        if process_index == 0:
            print("\nScoring model performance on validation data...\n")

        val_iter = iter(val_dl)
        val_shard_cursor = None
        local_val_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        local_val_steps = 0
        completed_val_shards = 0
        next_val_file_index = 0

        while True:
            (
                batch_ready,
                val_shard_cursor,
                closed_val_shards,
                loaded_val_shards,
                next_val_file_index,
            ) = (
                prepare_local_batch_buffer(
                    val_iter,
                    val_shard_cursor,
                    bsz,
                    seqlen,
                    data_sharding,
                    val_data_buf,
                    next_val_file_index,
                )
            )
            completed_val_shards += closed_val_shards

            if process_index == 0:
                for shard_name, shard_batch_count in loaded_val_shards:
                    print(
                        f"Loaded Validation Shard: {shard_name} | "
                        f"Indexed {shard_batch_count} batches"
                    )

            local_val_batch_flag = np.array(1 if batch_ready else 0, dtype=np.int32)
            all_val_batch_flags = np.asarray(
                process_allgather(local_val_batch_flag, tiled=False)
            )
            if int(np.min(all_val_batch_flags)) == 0:
                break

            curr_val_data = jnp.asarray(val_data_buf, dtype=jnp.int32, device=data_sharding)
            x = curr_val_data[:, :-1]
            y = curr_val_data[:, 1:]
            loss = val_step(model, x, y, segment_ids, freqs, None)
            local_val_loss_sum = local_val_loss_sum + loss
            local_val_steps += 1

        release_shard_tokens(val_shard_cursor)

        local_val_loss_sum = float(jax.block_until_ready(local_val_loss_sum))
        local_val_stats = np.array([local_val_loss_sum, float(local_val_steps)], dtype=np.float64)

        all_val_stats = np.asarray(process_allgather(local_val_stats, tiled=False))
        global_val_loss_sum = float(np.sum(all_val_stats[:, 0]))
        global_val_steps = int(np.sum(all_val_stats[:, 1]))

        if global_val_steps == 0:
            if process_index == 0:
                print(
                    "Skipping validation because no globally aligned "
                    "validation batches were available.\n"
                )
            continue

        if process_index == 0 and completed_val_shards > 0:
            print(f"Completed validation shards on host 0: {completed_val_shards}")

        avg_val_loss = global_val_loss_sum / global_val_steps
        improved = avg_val_loss < best_loss
        if improved:
            best_loss = avg_val_loss
            best_step = step
            es_patience_counter = 0
        else:
            es_patience_counter += 1

        if process_index == 0:
            print(f"last_val_loss : {last_val_loss:.4f}")
            print(f"curr_val_loss : {avg_val_loss:.4f}")
            print(f"Best loss     : {best_loss:.4f} at step {best_step}\n")
        last_val_loss = avg_val_loss

        local_stop_flag = np.array(
            1 if es_patience_counter > es_patience else 0, dtype=np.int32
        )
        all_stop_flags = np.asarray(process_allgather(local_stop_flag, tiled=False))
        training_complete = int(np.max(all_stop_flags)) == 1

        if training_complete and process_index == 0:
            print(
                f"\nEarly stopping triggered! No improvement for "
                f"{es_patience_counter} validation rounds."
            )
            print(f"Best loss : {best_loss:.4f} at step {best_step}")

    release_shard_tokens(train_shard_cursor)
    mngr.wait_until_finished()

    if process_index == 0:
        if step >= total_train_steps:
            print(f"\nReached maximum training steps  : {total_train_steps}")
        elif stopped_for_data_exhaustion:
            print(
                "\nStopped training because at least one host exhausted "
                "its local train data."
            )
        elif training_complete:
            print(
                f"\nEarly stopping triggered! No improvement for "
                f"{es_patience_counter} validation rounds."
            )

        print(f"Total number of shards consumed : {num_shards_used}")
        print(f"Total Tokens consumed          : {total_tokens_consumed:>9,}")
        print(f"Best loss                      : {best_loss:.4f} at step {best_step}")
        print("Finished checkpointing! Cleaned.")
    train_end_time = time.time()
    if process_index == 0:
        print(
            f"\nTotal time taken to train the model: {(train_end_time - train_start_time) / 60:.2f} minutes"
        )


if __name__ == "__main__":
    main()
