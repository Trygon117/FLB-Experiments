import torch
import torch.nn as nn

def train_model(model, sequence, window_config, optimizer=None, criterion=nn.CrossEntropyLoss(reduction='none'), layer_context=None, on_window=None):
    batch_size, seq_len = sequence.shape
    window_size = window_config["window_size"]
    stride = window_config["stride"]

    needed_len = window_size + 1

    assert seq_len >= needed_len, (
        f"Sequence length ({seq_len}) is too short. "
        f"Must be at least {needed_len} tokens for window + next token target."
    )

    total_loss = 0.0
    window_count = 0

    # Slide across the sequence
    for start in range(0, seq_len - window_size, stride):
        if optimizer is not None:
            optimizer.zero_grad()

        input = sequence[:, start : start + window_size]
        targets = sequence[:, start + 1 : start + window_size + 1]

        if layer_context is not None:
            outputs, layer_context = model.stream_model(input, layer_context, stride)
        else: 
            # Assume there is no layer_context
            outputs = model.stream_model(input, stride)

        raw_loss = criterion(
            outputs.reshape(-1, model.vocab_size),
            targets.reshape(-1)
        ).reshape(batch_size, -1)

        step_loss = raw_loss.mean()
        step_loss.backward()

        if on_window is not None:
            on_window({
                'model': model,
                'window_idx': window_count,
                'step_loss': step_loss.item(),
                'input': input,
                'outputs': outputs,
                'predictions': outputs,
                'targets': targets,
                'raw_loss': raw_loss.detach()
            })

        if optimizer is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += step_loss.item()
        window_count += 1

    avg_loss = total_loss / max(1, window_count)
    return avg_loss, layer_context