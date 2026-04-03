import jax
import grain
import hashlib
import orbax.checkpoint as ocp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P


def get_sharding_for_checkpoint(x, mesh):
    """Obtain shardings for the leaves of the pytree."""
    if hasattr(x, "ndim") and x.ndim == 0:
        return NamedSharding(mesh, P())
    if isinstance(x, jax.Array) and hasattr(x, "sharding"):
        from jax.sharding import SingleDeviceSharding

        # Ensure small optimizer leaves (e.g., Muon scalars/vectors) are replicated,
        # not left on a single device, to match param shardings during train_step.
        if isinstance(x.sharding, SingleDeviceSharding):
            return NamedSharding(mesh, P())
        return x.sharding
    else:
        return NamedSharding(mesh, P())


def load_checkpoint(mngr, step, model, optim_state, mesh, ds_iter=None):
    """Load the weights, the optimizer state, and (optional) data iterator state
    from a given checkpoint.

    Args:
        mngr: Checkpoint manager instance.
        step: The step from which the state has to be restored.
        model: Pytree  containing params.
        optim_state: Current optimizer state.
        mesh: Current mesh where the model and optimizer state is alive.
        ds_iter (optional): The data iterator whose state is to be restored from
            this checkpoint. Defaults to `None`.

    Returns:
        Tuple of (Restored weights, restored optim_state, restored ds_iter)
    """

    params_item, params_transforms = model, None
    optim_item, optim_transforms = optim_state, None
    params_restore_args = jax.tree.map(
        lambda s: ocp.ArrayRestoreArgs(sharding=get_sharding_for_checkpoint(s, mesh)),
        model,
    )
    optim_restore_args = jax.tree.map(
        lambda s: ocp.ArrayRestoreArgs(sharding=get_sharding_for_checkpoint(s, mesh)),
        optim_state,
    )
    if ds_iter is not None:
        restore_items = ocp.args.Composite(
            params=ocp.args.PyTreeRestore(
                item=params_item,
                transforms=params_transforms,
                restore_args=params_restore_args,
            ),
            optim_state=ocp.args.PyTreeRestore(
                item=optim_item,
                transforms=optim_transforms,
                restore_args=optim_restore_args,
            ),
            ds=grain.checkpoint.CheckpointRestore(ds_iter),
        )
        restored = mngr.restore(step, args=restore_items)
        print(
            f"Restoring model, optim_state, and data_iter from step {step} is complete!"
        )
        return restored.params, restored.optim_state, restored.ds
    else:
        restore_items = ocp.args.Composite(
            params=ocp.args.PyTreeRestore(
                item=params_item,
                transforms=params_transforms,
                restore_args=params_restore_args,
            ),
            optim_state=ocp.args.PyTreeRestore(
                item=optim_item,
                transforms=optim_transforms,
                restore_args=optim_restore_args,
            ),
        )
        restored = mngr.restore(step, args=restore_items)
        print(f"Restoring model and optim_state from step {step} is complete!")
        return restored.params, restored.optim_state


def load_weights_from_checkpoint(path, sharding):
    """Load weights of the model from a given checkpoint.

    Args:
        path: Path to the saved pytree checkpointed during training.
        sharding: The sharding info obtained from the current state of the pytree.

    Returns:
        PyTree with weights loaded from the given checkpoint.
    """

    print(f"Restoring checkpoint from: {path}")
    item, transforms = sharding, None
    restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=item, transforms=transforms, restore_args=restore_args
            ),
        )


def tree_path_to_str(path):
    """Converts flattened tree path with leaves into a string.

    Args:
        path: leaf path
    Returns:
        A combined string representation of the path
    """

    parts = []
    for p in path:
        if hasattr(p, "key"):
            parts.append(str(p.key))
        elif hasattr(p, "name"):
            parts.append(str(p.name))
        elif hasattr(p, "idx"):
            parts.append(str(p.idx))
        else:
            parts.append(str(p))
    return "/".join(parts)


def extract_shapes_and_dtypes(tree):
    """Extracts the shapes and dtypes of leaves (arrays) from a pytree.

    Args:
        tree: A PyTree instance
    Returns:
        A dictionary of leaves where the value corresponding to a leaf
        contains the shape and dtype of that leaf. A leaf here represents
        a jax array (e.g. model weights).
    """

    path_vals, _ = jax.tree_util.tree_flatten_with_path(tree)
    flattened = {}
    for path, leaf in path_vals:
        if leaf is None:
            continue
        if not hasattr(leaf, "shape"):
            continue
        flattened[tree_path_to_str(path)] = leaf
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in flattened.items()}


def get_schema_hash(tree):
    """Extracts the schema (shapes and dtypes) of a pytree, and calculates
    a hash of it."""
    schema = extract_shapes_and_dtypes(tree)
    entries = [f"{k}:{shape}:{dtype}" for k, (shape, dtype) in schema.items()]
    entries.sort()
    blob = "\n".join(entries)
    return hashlib.sha256(blob.encode()).hexdigest()


def pytrees_equal(pytree1, pytree2):
    return extract_shapes_and_dtypes(pytree1) == extract_shapes_and_dtypes(pytree2)


def print_diff(params_struct, ckpt_struct):
    """Prints the difference between two abstract pytrees structures."""

    params_schema = extract_shapes_and_dtypes(params_struct)
    ckpt_schema = extract_shapes_and_dtypes(ckpt_struct)

    param_schema_set = set(params_schema)
    ckpt_schema_set = set(ckpt_schema)

    missing = sorted(param_schema_set - ckpt_schema_set)
    extra = sorted(ckpt_schema_set - param_schema_set)
    mismatch = sorted(
        k
        for k in params_schema.keys() & ckpt_schema.keys()
        if params_schema[k] != ckpt_schema[k]
    )

    if not missing and not extra and not mismatch:
        print("Pytree match!")
        return True

    if missing:
        print("\nMissing in checkpoint:")
        for key in missing:
            print(" ", key, params_schema[key])

    if extra:
        print("\nExtra in checkpoint:")
        for key in extra:
            print(" ", key, ckpt_schema[key])

    if mismatch:
        print("\nShape or dtype mismatch found!")
        for key in mismatch:
            print("Key: ", key)
            print("    Param schema: ", params_schema[key])
            print("    Ckpt  schema: ", ckpt_schema[key], "\n")


def validate_checkpoint(params_struct, ckpt_struct, strict=True):
    """Checks if the current param pytree is valid for a given checkpoint.

    Args:
        param_struct: Abstract pytree of the current params
        ckpt_struct : Abstract pytree of the checkpoint
    Returns:
        True/False depending on whether the param abstract pytree matches
        with the abstract pytree of the given checkpoint.
    """

    model_hash = get_schema_hash(params_struct)
    ckpt_hash = get_schema_hash(ckpt_struct)
    return model_hash == ckpt_hash


# There is a bug in orbax where it can silently restore a pytree partially even
# when strict is enabled. It will likely be fixed in the v1 API. For now, we will
# use this custom function along with some utilities to ensure that we only load
# checkpoint if the pytree structure, data types and shapes matches exactly!
def load_weights_from_checkpoint_with_validation(path, params, sharding, strict=True):
    print(f"Reading checkpoint metadata from: {path}")
    with ocp.PyTreeCheckpointer() as ckptr:
        ckpt_metadata = ckptr.metadata(path)

    params_struct = jax.tree.map(ocp.utils.to_shape_dtype_struct, params)
    # ckpt_struct = jax.tree.map(ocp.utils.to_shape_dtype_struct, ckpt_metadata.item_metadata.tree)
    ckpt_struct = ckpt_metadata.item_metadata.tree

    print("Validating params structure and checkpoint being loaded...\n")
    is_valid = validate_checkpoint(params_struct, ckpt_struct)

    if not is_valid:
        if strict:
            print_diff(params_struct, ckpt_struct)
            raise RuntimeError(
                "\nThe model structure does not match with the checkpoint being loaded!"
            )
        else:
            print("Validation failed...")
            raise RuntimeError(
                "\nThe model structure does not match with the checkpoint being loaded!"
            )

    print(f"Restoring params from: {path}")
    item, transforms = sharding, None
    restore_args = jax.tree.map(lambda s: ocp.ArrayRestoreArgs(sharding=s), sharding)
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=item, transforms=transforms, restore_args=restore_args
            ),
        )


def load_optim_state_from_checkpoint(path, optim_state, mesh):
    """Load optimizer state from a given checkpoint.

    Args:
        path: Path to the saved pytree checkpointed during training.
        optim_state: The current optimizer state.
        mesh: Mesh of the current optimizer state

    Returns:
        PyTree with optim state loaded from the given checkpoint.
    """

    print(f"Restoring optimizer state from checkpoint: {path}")
    optim_item, optim_transforms = optim_state, None
    optim_restore_args = jax.tree.map(
        lambda s: ocp.ArrayRestoreArgs(sharding=get_sharding_for_checkpoint(s, mesh)),
        optim_state,
    )
    with ocp.PyTreeCheckpointer() as ckptr:
        return ckptr.restore(
            path,
            args=ocp.args.PyTreeRestore(
                item=optim_state,
                transforms=optim_transforms,
                restore_args=optim_restore_args,
            ),
        )
