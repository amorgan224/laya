import argparse
import os
import torch

from laya.agent import Agent

def export_to_onnx(model_id_or_path: str, output_path: str):
    print(f"Loading PyTorch Agent from: {model_id_or_path}")
    agent = Agent(model_id_or_path, compile=False, device="cpu")
    
    print("Creating dummy input tensors...")
    # 1. Dummy tensors for tracing
    # (batch_size=1, seq_len=16)
    dummy_input_ids = torch.randint(0, 100, (1, 16), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, 16), dtype=torch.long)
    
    # (batch_size=1, num_markers=2)
    dummy_marker_pos = torch.tensor([[1, 5]], dtype=torch.long)
    dummy_marker_mask = torch.tensor([[True, True]], dtype=torch.bool)
    
    # (batch_size=1)
    dummy_qtype = torch.tensor([0], dtype=torch.long)
    
    inputs = (
        dummy_input_ids,
        dummy_attention_mask,
        dummy_marker_pos,
        dummy_marker_mask,
        dummy_qtype,
    )

    # 2. Define dynamic axes so the model can accept variable batch sizes and sequence lengths
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"},
        "marker_pos": {0: "batch_size", 1: "num_markers"},
        "marker_mask": {0: "batch_size", 1: "num_markers"},
        "qtype": {0: "batch_size"},
        "logits": {0: "batch_size", 1: "num_markers"},
        "act_logits": {0: "batch_size"},
    }

    input_names = [
        "input_ids",
        "attention_mask",
        "marker_pos",
        "marker_mask",
        "qtype",
    ]
    
    output_names = ["logits", "act_logits"]

    print(f"Exporting to {output_path} (this may take a minute)...")
    
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # We must detach the encoder because ONNX export runs the model in trace mode.
    # The `detach_encoder` flag in forward() just detaches the hidden state gradient, 
    # but we don't even need to pass it since kwargs are ignored by tracing.
    
    torch.onnx.export(
        agent.model,
        inputs,
        output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    
    print(f"Successfully exported ONNX model to: {output_path}")

F = TypeVar("F", bound=Callable)

def retry_with_backoff(
    max_attempts: int = 3,
    exceptions: Tuple[type[BaseException], ...] = (Exception,),
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    on_retry: Optional[Callable[[BaseException, int], None]] = None
) -> Callable[[F], F]:
    """
    Retry a function with exponential backoff.

    Args:
        max_attempts: Total number of attempts before re-raising the last error.
        exceptions: Tuple of exceptions to catch and retry on.
        base_delay: Initial delay in seconds.
        max_delay: Cap on delay between retries.
        jitter: If True, add random jitter to avoid thundering herd.
        on_retry: Optional callback invoked with (exception, attempt_number).
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Optional[BaseException] = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    if attempt == max_attempts:
                        break

                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay = random.uniform(0, delay)

                    logging.warning(
                        "%s failed (attempt %d/%d): %s. Retrying in %.2fs...",
                        func.__name__, attempt, max_attempts, exc, delay
                    )

                    if on_retry:
                        on_retry(exc, attempt)

                    time.sleep(delay)

            raise last_exception  # type: ignore

        return wrapper  # type: ignore

    return decorator


# ─── Example usage ───

@retry_with_backoff(
    max_attempts=4,
    exceptions=(ConnectionError, TimeoutError),
    base_delay=0.5
)
def call_flaky_api(endpoint: str) -> dict:
    """Simulate a call that sometimes fails."""
    if random.random() < 0.7:
        raise ConnectionError(f"Failed to reach {endpoint}")
    return {"status": "ok", "endpoint": endpoint}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = call_flaky_api("https://api.example.com/data")
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a Laya model to ONNX format")
    parser.add_argument("--model", type=str, default="convaiinnovations/laya", help="HuggingFace Hub ID or local path")
    parser.add_argument("--output", type=str, default="laya.onnx", help="Output path for the ONNX file")
    args = parser.parse_args()
    
    export_to_onnx(args.model, args.output)
