import torch
import torch.nn as nn

def train_baseline_model(model, optimizer, get_batch_fn, batch_size, seq_len, window_config, vocab_size, num_steps=100, tracker=None, max_grad_norm=1.0, log_interval=100):
    """
    Trains a baseline model across sliding windows with identical telemetry to FLB.
    """
    criterion = nn.CrossEntropyLoss(reduction='none')
    window_size = window_config["window_size"]
    stride = window_config["stride"]

    for step in range(1, num_steps + 1):
        batch = get_batch_fn(batch_size=batch_size, seq_len=seq_len)
        total_loss = 0.0
        window_count = 0

        # Slide across windows exactly matching FLB
        for start in range(0, seq_len - window_size, stride):
            optimizer.zero_grad()

            input_window = batch[:, start : start + window_size]
            target_window = batch[:, start + 1 : start + window_size + 1]

            # Forward pass
            outputs = model(input_window)

            # Unreduced per-token loss
            raw_loss = criterion(
                outputs.reshape(-1, vocab_size),
                target_window.reshape(-1)
            ).reshape(batch_size, -1)

            loss = raw_loss.mean()
            loss.backward()

            # Record telemetry if tracker provided
            if tracker is not None:
                tracker({
                    'model': model,
                    'window_idx': window_count,
                    'step_loss': loss.item(),
                    'input': input_window,
                    'outputs': outputs,
                    'predictions': outputs,
                    'targets': target_window,
                    'raw_loss': raw_loss.detach()
                })

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            window_count += 1

        if step % log_interval == 0 or step == 1:
            avg_loss = total_loss / max(1, window_count)
            print(f"Step {step:04d} | Window Avg Loss {avg_loss:.4f}")

    if tracker is not None:
        tracker.save()