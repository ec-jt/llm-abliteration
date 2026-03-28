import gc
import importlib
import os
import sys
import torch
import torch.distributed as dist
from argparse import ArgumentParser
from tqdm import tqdm
import transformers.utils as tf_utils
import transformers.utils.import_utils as tf_import_utils
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoTokenizer
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils.data import load_data
from utils.models import has_tied_weights
from utils.clip import magnitude_clip
from utils.device import clear_device_cache, get_preferred_device, resolve_device_map, synchronize_device


# Compatibility shim for some remote model repos that still import the old helper
# name from transformers.utils.
if not hasattr(tf_utils, "is_flash_attn_greater_or_equal_2_10"):
    def _is_flash_attn_greater_or_equal_2_10() -> bool:
        if hasattr(tf_utils, "is_flash_attn_greater_or_equal"):
            try:
                return tf_utils.is_flash_attn_greater_or_equal("2.10")
            except TypeError:
                # Older/newer signatures can differ; conservative fallback.
                return bool(tf_utils.is_flash_attn_greater_or_equal())
        return False

    tf_utils.is_flash_attn_greater_or_equal_2_10 = _is_flash_attn_greater_or_equal_2_10

# Compatibility shim for remote repos expecting the older import_utils helper.
if not hasattr(tf_import_utils, "is_torch_fx_available"):
    def _is_torch_fx_available() -> bool:
        if not hasattr(tf_import_utils, "is_torch_available"):
            return False
        if not tf_import_utils.is_torch_available():
            return False
        try:
            import torch.fx  # noqa: F401
            return True
        except Exception:
            return False

    tf_import_utils.is_torch_fx_available = _is_torch_fx_available


def welford_gpu_batched_multilayer_float32(
    formatted_prompts: list[str],
    desc: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    layer_indices: list[int],
    position: int = 1,
    batch_size: int = 1,
    clip: float = 1.0,
    processor = None,  # Add processor parameter
    is_vision_model: bool = False,  # Add flag for vision models
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> dict[int, torch.Tensor]:
    text_config = model.config
    if hasattr(text_config, "text_config"):
        text_config = text_config.text_config
    hidden_size = getattr(text_config, "hidden_size", None)

    max_tokens = position

    sums = {layer_idx: None for layer_idx in layer_indices}
    counts = {layer_idx: 0 for layer_idx in layer_indices}
    show_progress = rank == 0

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    if is_vision_model and processor is not None:
        processor.tokenizer.padding_side = 'left'

    for i in tqdm(range(0, len(formatted_prompts), batch_size), desc=desc, disable=not show_progress):
        batch_prompts = formatted_prompts[i:i+batch_size]

        if is_vision_model and processor is not None:
            # For vision models, use the processor with text-only input
            batch_encoding = processor(
                text=batch_prompts,
                return_tensors="pt",
                padding=True,
            )
        else:
            # For text-only models, use the tokenizer
            batch_encoding = tokenizer(
                batch_prompts,
                padding=True,
                return_tensors="pt",
            )
        
        batch_input = batch_encoding['input_ids'].to(model.device)
        batch_mask = batch_encoding['attention_mask'].to(model.device)

        # Use generate to get hidden states at the first generated token position
        raw_output = model.generate(
            batch_input,
            attention_mask=batch_mask,
            max_new_tokens=max_tokens,
            return_dict_in_generate=True,
            output_hidden_states=True,
            pad_token_id=tokenizer.eos_token_id,
#            do_sample=False,                       # disable sampling
#            top_k=None,
#            top_p=None,
#            cache_implementation=None,
        )
        
        #last_non_pad = batch_mask.sum(dim=1) - 1  # shape: (batch,)
        del batch_input, batch_mask
        #hidden_states = raw_output.hidden_states[max_tokens-1] # Generation step
        hidden_states = [
            layer_tensor.detach().clone() 
            for layer_tensor in raw_output.hidden_states[-1]
        ]
        del raw_output
        #last_non_pad = last_non_pad.to(hidden_states.device)

        # Process layers with Welford in float32
        for layer_idx in layer_indices:
            # Cast to float32 for accumulation
            # Index each sample at its own last non-pad position
#            current_hidden = hidden_states[layer_idx][
#                torch.arange(hidden_states[layer_idx].size(0), device=hidden_states[layer_idx].device),
#                last_non_pad,
#                :
#            ].float() # (batch, hidden)
            # only examine state after last generated position
            current_hidden = hidden_states[layer_idx][:, -1, :].double()
            #current_hidden = hidden_states[layer_idx][:, pos, :].float()
            if (clip < 1.0):
                current_hidden = magnitude_clip(current_hidden, clip)

            batch_size_actual = current_hidden.size(dim=0)
            if sums[layer_idx] is None:
                sums[layer_idx] = current_hidden.double().sum(dim=0)
            else:
                sums[layer_idx] += current_hidden.double().sum(dim=0)

            counts[layer_idx] += batch_size_actual
            del current_hidden

        del hidden_states
        #del last_non_pad
        clear_device_cache()

    # Reduce global sums/counts across ranks when distributed, then compute means.
    return_dict = {}
    for layer_idx in layer_indices:
        local_sum = sums[layer_idx]
        if local_sum is None:
            if hidden_size is None:
                raise ValueError("Could not infer hidden size for distributed reduction")
            reduce_device = model.device if hasattr(model, "device") else "cpu"
            local_sum = torch.zeros(hidden_size, dtype=torch.float64, device=reduce_device)

        local_count_t = torch.tensor(float(counts[layer_idx]), dtype=torch.float64, device=local_sum.device)

        if distributed and world_size > 1:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_count_t, op=dist.ReduceOp.SUM)

        global_count = int(local_count_t.item())
        if global_count == 0:
            raise ValueError(f"No prompts available to measure for layer {layer_idx}")

        mean = local_sum / float(global_count)
        return_dict[layer_idx] = mean.to(device="cpu")

    del sums
    clear_device_cache()
    return return_dict


def init_distributed(
    enabled: bool,
    backend: str | None = None,
    master_addr: str | None = None,
    master_port: int | None = None,
    rank: int | None = None,
    world_size: int | None = None,
) -> tuple[bool, int, int, int]:
    if not enabled:
        return False, 0, 1, 0

    if master_addr:
        os.environ["MASTER_ADDR"] = master_addr
    if master_port:
        os.environ["MASTER_PORT"] = str(master_port)
    if rank is not None:
        os.environ["RANK"] = str(rank)
    if world_size is not None:
        os.environ["WORLD_SIZE"] = str(world_size)

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in this PyTorch build")

    inferred_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if not dist.is_initialized():
        dist.init_process_group(backend=inferred_backend, init_method="env://")

    rank_v = dist.get_rank()
    world_size_v = dist.get_world_size()
    local_rank_v = int(os.environ.get("LOCAL_RANK", "0"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank_v)

    return True, rank_v, world_size_v, local_rank_v


def shard_list_for_rank(items: list[str], rank: int, world_size: int) -> list[str]:
    if world_size <= 1:
        return items
    return items[rank::world_size]

def format_chats(
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt_list: list[str],
    processor = None,
):
    # Use processor's tokenizer if available, otherwise use tokenizer directly
    actual_tokenizer = processor.tokenizer if processor is not None else tokenizer
    
    result_formatted = [
        actual_tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": inst}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for inst in prompt_list
    ]
    return result_formatted

def compute_refusals(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    harmful_list: list[str],
    harmless_list: list[str],
    projected: bool = False,
    inference_batch_size: int = 32,
    clip: float = 1.0,
    processor = None,  # processor parameter
    is_vision_model: bool = False,  # flag for vision models
    token2: bool = False, # measure at second token instead of first
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> torch.Tensor:
    # dtype = model.dtype
    if hasattr(model, "language_model"):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, "language_model"):
            layer_base = layer_base.language_model
    num_layers = len(layer_base.layers)

    pos = 1
    if token2:
        pos = 2

    focus_layers = range(num_layers) # sweep all layers

    harmful_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmful_list, processor=processor)
    harmful_means = welford_gpu_batched_multilayer_float32(
        harmful_formatted, "Generating harmful outputs", model, tokenizer,
        focus_layers, pos, inference_batch_size, clip, processor, is_vision_model,
        distributed, rank, world_size,
    )
    clear_device_cache()
    del harmful_formatted
    harmless_formatted = format_chats(tokenizer=tokenizer, prompt_list=harmless_list, processor=processor)
    harmless_means = welford_gpu_batched_multilayer_float32(
        harmless_formatted, "Generating harmless outputs", model, tokenizer, 
        focus_layers, pos, inference_batch_size, clip, processor, is_vision_model,
        distributed, rank, world_size,
    )
    del harmless_formatted

    results = {}
    results["layers"] = num_layers

    # Keep all results in 32-bit float for analysis/ablation
    for layer in tqdm(focus_layers, desc="Compiling layer measurements", disable=rank != 0):
        harmful_mean = harmful_means[layer]
        results[f'harmful_{layer}'] = harmful_mean.to(dtype=model.dtype)
        harmless_mean = harmless_means[layer]
        results[f'harmless_{layer}'] = harmless_mean.to(dtype=model.dtype)

        harmful_d = harmful_mean.double()
        harmless_d = harmless_mean.double()

        # Compute raw difference of means in float64 to avoid cancellation at high cosine similarity.
        # Saved unnormalized — normalization is deferred to the ablation phase.
        # Note: once unit-normalized, harmful_hat - harmless_hat is exactly the normal of the
        # Householder reflector that maps harmless_hat onto harmful_hat, giving this direction
        # a clean geometric justification beyond naive contrastive difference of means.

        refusal_dir = harmful_d - harmless_d
        results[f'refuse_{layer}'] = refusal_dir.to(dtype=model.dtype)

        # Householder-inspired alternative of computing difference after normalization
        # Unfortunately, it is inferior numerically even if analytically correct
        #harmful_hat = torch.nn.functional.normalize(harmful_d, dim=0)
        #harmless_hat = torch.nn.functional.normalize(harmless_d, dim=0)
        #refusal_dir = harmful_hat - harmless_hat

        if projected: # semantic meaning: preserve activations along the harmless direction
            # Compute Gram-Schmidt second orthogonal vector/direction to remove harmless direction interference from refusal direction
            # Two-pass Gram-Schmidt — second pass catches residual from float cancellation
            harmless_hat = torch.nn.functional.normalize(harmless_d, dim=0)
            refusal_dir = refusal_dir - (refusal_dir @ harmless_hat) * harmless_hat
            refusal_dir = refusal_dir - (refusal_dir @ harmless_hat) * harmless_hat

        refusal_dir = torch.nn.functional.normalize(refusal_dir, dim=0)

        results[f'refusenorm_{layer}'] = refusal_dir.to(dtype=model.dtype)

    clear_device_cache()
    gc.collect()
    return results


def clean_up() -> None:
    """
    Release VRAM/RAM after measurement is complete.

    Call this after deleting model/tokenizer/results in your code:
        del model, tokenizer, processor, results
        clean_up()

    Note: Callers must delete their own references to objects before calling
    this function. Python's scoping rules mean we cannot delete caller's
    variables from within a function.
    """
    gc.collect()
    synchronize_device()
    clear_device_cache()
    gc.collect()  # Second pass for any refs broken by cache clear
    print("Memory cleared successfully.")


def debug_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            t = output[0]
        else:
            t = output
        inp = input[0] if isinstance(input, tuple) else input
        inp_max = inp.abs().max().item()
        out_max = t.abs().max().item() if not torch.isnan(t).any() else float('nan')
        print(f"Layer {name}: input_max={inp_max:.4f} | output_max={out_max:.4f}")
    return hook



if __name__ == "__main__":
    parser = ArgumentParser(description="Measure models for analysis and abliteration")
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        required=True,
        help="Local model directory or HuggingFace model ID",
    )
    parser.add_argument(
        "--quant-measure", "-q",
        type=str,
        choices=["4bit", "8bit"],
        default=None,
        help="Perform measurement using 4bit or 8bit bitsandbytes quant"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size during inference/calibration; default 32, stick to powers of 2 (higher will use more VRAM)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        required=True,
        help="Output file for measurements"
    )
    parser.add_argument(
        "--clip",
        type=float,
        default=1.0,
        help="Fraction of prompt activation to clip by magnitude",
    )
    parser.add_argument(
        "--flash-attn",
        action="store_true",
        default=False,
        help="Use Flash Attention 2"
    )
    parser.add_argument(
        "--data-harmful",
        type=str,
        default=None,
        help="Harmful prompts file"
    )
    parser.add_argument(
        "--data-harmless",
        type=str,
        default=None,
        help="Harmless prompts file"
    )
    parser.add_argument(
        "--deccp",
        action="store_true",
        default=False,
        help="For Chinese models, add topics to harmful prompts",
    )
    parser.add_argument(
        "--projected",
        action="store_true",
        default=False,
        help="Remove projection along harmless direction from contrast direction",
    )
    parser.add_argument(
        "--token2",
        action="store_true",
        default=False,
        help="Measure after second token instead of after first token",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Allow execution of custom model/tokenizer code from model repository",
    )
    parser.add_argument(
        "--allow-model-compression",
        action="store_true",
        default=False,
        help="Allow model repo quantization/compression metadata when --quant-measure is not set",
    )
    parser.add_argument(
        "--dist",
        action="store_true",
        default=False,
        help="Enable torch.distributed multi-process measurement",
    )
    parser.add_argument(
        "--dist-backend",
        type=str,
        choices=["nccl", "gloo", "mpi"],
        default=None,
        help="Distributed backend (default: nccl on CUDA, else gloo)",
    )
    parser.add_argument(
        "--master-addr",
        type=str,
        default=None,
        help="MASTER_ADDR override for multi-node torchrun",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="MASTER_PORT override for multi-node torchrun",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="RANK override when not provided by torchrun",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="WORLD_SIZE override when not provided by torchrun",
    )

    args = parser.parse_args()

    assert (
        isinstance(args.model, str)
        and
        isinstance(args.output, str)
    )

    qbit = args.quant_measure

    dist_enabled, rank, world_size, local_rank = init_distributed(
        enabled=args.dist,
        backend=args.dist_backend,
        master_addr=args.master_addr,
        master_port=args.master_port,
        rank=args.rank,
        world_size=args.world_size,
    )

    torch.inference_mode()
    torch.set_grad_enabled(False)

    device = get_preferred_device()
    device_map = resolve_device_map()
    if dist_enabled and device == "cuda":
        # One process per GPU under torchrun; avoid each rank claiming all local GPUs.
        device_map = {"": local_rank}
        if rank == 0:
            print(f"Distributed mode enabled: world_size={world_size}, backend={dist.get_backend()}")
    elif dist_enabled and rank == 0:
        print(f"Distributed mode enabled on {device}: world_size={world_size}, backend={dist.get_backend()}")

    model = args.model
    model_config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=args.trust_remote_code,
    )

    # Loading policy:
    # - If --quant-measure is set, preserve model quantization metadata and normalize
    #   quantization_config=None -> {} for remote repos that otherwise crash.
    # - Otherwise, default to full-precision by removing model-provided
    #   quantization/compression metadata, unless explicitly allowed.
    if qbit:
        if hasattr(model_config, "quantization_config") and getattr(model_config, "quantization_config") is None:
            setattr(model_config, "quantization_config", {})
    elif not args.allow_model_compression:
        for _cfg_attr in ("quantization_config", "compression_config"):
            if hasattr(model_config, _cfg_attr):
                try:
                    delattr(model_config, _cfg_attr)
                except Exception:
                    setattr(model_config, _cfg_attr, None)

    model_type = getattr(model_config,"model_type")

    # Get the precision/dtype from config, with proper fallback
    if hasattr(model_config, "torch_dtype") and model_config.torch_dtype is not None:
        precision = model_config.torch_dtype
    elif hasattr(model_config, "dtype") and model_config.dtype is not None:
        precision = model_config.dtype
    else:
        # Fallback to bfloat16 on CUDA (if supported), otherwise float32 on MPS/CPU, float16 on CUDA
        if device == "cuda" and torch.cuda.is_bf16_supported():
            precision = torch.bfloat16
        elif device == "cuda":
            precision = torch.float16
        else:
            precision = torch.float32

    has_vision = False
    if hasattr(model_config, "vision_config"):
        has_vision = True
    model_loader = AutoModelForCausalLM

    quant_config = None

    if device == "mps" and qbit:
        print("BitsAndBytes quantization is not supported on MPS; disabling requested quantization.")
        qbit = None

    # Quantization is opt-in only: no automatic detection from model config.
    # Full-precision loading remains the default unless --quant-measure is set.
    if qbit:
        if rank == 0:
            print(f"Load mode: explicit bitsandbytes {qbit} quantization (--quant-measure).")
    elif args.allow_model_compression:
        if rank == 0:
            print("Load mode: allowing model-defined quantization/compression metadata.")
    else:
        if rank == 0:
            print("Load mode: forcing full-precision load; ignoring model-defined quantization/compression metadata.")

    # Convert string dtype to torch dtype if needed
    if isinstance(precision, str):
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        precision = dtype_map.get(
            precision,
            torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32,
        )

    if qbit == "4bit":
        print(f"Using compute dtype from quant config: {precision}")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=precision,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4", # better for QLoRA
        )
    elif qbit == "8bit":
        quant_config = BitsAndBytesConfig(
            load_in_8bit=True,
#            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_has_fp16_weight=False,
#            llm_int8_threshold=6.0,
        )    

    if isinstance(args.data_harmful, str):
        harmful_list = load_data(args.data_harmful)
    else:
        harmful_list = load_data("./data/harmful.parquet")
    if isinstance(args.data_harmless, str):
        harmless_list = load_data(args.data_harmless)
    else:
        harmless_list = load_data("./data/harmless.parquet")

    if args.deccp:
        from datasets import load_dataset
        deccp_list = load_dataset("augmxnt/deccp", split="censored")
        harmful_list += deccp_list["text"]

    if dist_enabled and world_size > 1:
        harmful_list = shard_list_for_rank(harmful_list, rank, world_size)
        harmless_list = shard_list_for_rank(harmless_list, rank, world_size)
        if rank == 0:
            print("Sharding prompt datasets across distributed ranks for data-parallel measurement.")

    # Kimi-K2.5 remote code may default parts of the model (e.g., vision tower)
    # to flash_attention_2, which can fail on unsupported submodules. Unless
    # explicitly requested, force eager attention for widest compatibility.
    attn_impl = "flash_attention_2" if args.flash_attn and device == "cuda" else "eager"
    if attn_impl == "eager":
        setattr(model_config, "_attn_implementation", "eager")
        if hasattr(model_config, "text_config") and getattr(model_config, "text_config") is not None:
            setattr(model_config.text_config, "_attn_implementation", "eager")
        if hasattr(model_config, "vision_config") and getattr(model_config, "vision_config") is not None:
            setattr(model_config.vision_config, "_attn_implementation", "eager")

    # Kimi-K2.5 remote modeling bug workaround:
    # MoonViT3dEncoder.__init__ references self.use_deterministic_attn before
    # setting it. Providing a class default prevents AttributeError while keeping
    # behavior deterministic=False unless remote code overrides it.
    if args.trust_remote_code:
        try:
            kimi_mod = importlib.import_module(
                "transformers_modules.Kimi_hyphen_K2_dot_5.modeling_kimi_k25"
            )
            if hasattr(kimi_mod, "MoonViT3dEncoder") and not hasattr(kimi_mod.MoonViT3dEncoder, "use_deterministic_attn"):
                setattr(kimi_mod.MoonViT3dEncoder, "use_deterministic_attn", False)
        except Exception:
            # Best effort only; if module path differs we continue without patch.
            pass

    def _load_model_with_current_loader():
        return model_loader.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            config=model_config,
            dtype=precision,
            low_cpu_mem_usage=True,
            device_map=device_map,
            attn_implementation=attn_impl,
            **({"quantization_config": quant_config} if quant_config is not None else {}),
        )

    def _patch_one_kimi_module(mod) -> bool:
        patched = False

        encoder_cls = getattr(mod, "MoonViT3dEncoder", None)
        if encoder_cls is not None and not getattr(encoder_cls, "_roo_patched_use_det", False):
            original_init = encoder_cls.__init__

            def _patched_init(self, *init_args, **init_kwargs):
                self.use_deterministic_attn = False
                return original_init(self, *init_args, **init_kwargs)

            encoder_cls.__init__ = _patched_init
            encoder_cls._roo_patched_use_det = True
            patched = True

        kimi_cls = getattr(mod, "KimiK25ForConditionalGeneration", None)
        if kimi_cls is not None and not getattr(kimi_cls, "_roo_patched_tie_weights", False):
            original_tie_weights = kimi_cls.tie_weights

            def _patched_tie_weights(self, *tw_args, **tw_kwargs):
                # transformers>=5.4 may pass recompute_mapping kwarg.
                return original_tie_weights(self)

            kimi_cls.tie_weights = _patched_tie_weights
            kimi_cls._roo_patched_tie_weights = True
            patched = True

        return patched

    def _patch_kimi_remote_code_compat() -> bool:
        patched = False

        # Try canonical dynamic-module path first.
        try:
            mod = importlib.import_module("transformers_modules.Kimi_hyphen_K2_dot_5.modeling_kimi_k25")
            patched = _patch_one_kimi_module(mod) or patched
        except Exception:
            pass

        # Also patch any already-loaded copies.
        for mod_name, mod in list(sys.modules.items()):
            if mod_name.endswith("modeling_kimi_k25"):
                patched = _patch_one_kimi_module(mod) or patched

        return patched

    # Prefer text-only loader first; some configs expose vision_config but are not
    # registered under AutoModelForImageTextToText. Retry with compatibility patches
    # when remote code expects older transformers internals.
    load_error = None
    for _ in range(3):
        try:
            model = _load_model_with_current_loader()
            load_error = None
            break
        except Exception as e:
            load_error = e
            err = str(e)

            if isinstance(e, ValueError) and has_vision:
                if rank == 0:
                    print(f"Vision/text loader mismatch ({e}); retrying with AutoModelForCausalLM.")
                has_vision = False
                model_loader = AutoModelForCausalLM
                continue

            needs_kimi_patch = (
                "use_deterministic_attn" in err
                or "tie_weights() got an unexpected keyword argument 'recompute_mapping'" in err
            )
            if needs_kimi_patch and _patch_kimi_remote_code_compat():
                if rank == 0:
                    print("Applied Kimi remote-code compatibility patches; retrying model load.")
                continue

            break

    if load_error is not None:
        raise load_error
    model.requires_grad_(False)
    if has_tied_weights(model_type):
        model.tie_weights()

    # point to base of language model
    if hasattr(model, "language_model"):
        layer_base = model.language_model.model
    else:
        layer_base = model.model
        if hasattr(layer_base, "language_model"):
            layer_base = layer_base.language_model


#    for i, layer in enumerate(layer_base.layers):
#        layer.register_forward_hook(debug_hook(i))

    #print(layer_base.embed_tokens.weight.dtype)
    #print(layer_base.config.hidden_size) 

    if qbit == "4bit": # stabilize for Gemma 3, possibly other models
        layer_base.embed_tokens = layer_base.embed_tokens.to(precision)
        layer_base.norm = layer_base.norm.to(precision)
    # Gemma 3 still needs Winsorization to not explode!

    # Load processor for vision models, tokenizer for text-only models
    processor = None
    if has_vision:
        try:
            processor = AutoProcessor.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code,
                device_map=device_map,
                padding=True,
            )
            tokenizer = processor.tokenizer
            if rank == 0:
                print("Loaded processor for vision model")
        except (IndexError, Exception) as e:
            # If processor loading fails, fall back to tokenizer only
            if rank == 0:
                print(f"Could not load processor ({e}), falling back to tokenizer only")
            has_vision = False
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=args.trust_remote_code,
                device_map=device_map,
                padding=True,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=args.trust_remote_code,
            device_map=device_map,
            padding=True,
        )

    if rank == 0:
        print("Computing refusal information...")
    results = {}
    results = compute_refusals(
        model=model,
        tokenizer=tokenizer,
        harmful_list=harmful_list,
        harmless_list=harmless_list,
        projected=args.projected,
        inference_batch_size=args.batch_size,
        clip=args.clip,
        processor=processor,
        is_vision_model=has_vision,
        token2=args.token2,
        distributed=dist_enabled,
        rank=rank,
        world_size=world_size,
    )

    if dist_enabled:
        dist.barrier()

    if rank == 0:
        print(f"Saving refusal information to {args.output}...")
        torch.save(results, args.output)

    # Release VRAM so next measurement can start immediately
    if rank == 0:
        print("Unloading model and clearing memory...")
    del model, tokenizer, processor, results
    clean_up()

    if dist_enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
