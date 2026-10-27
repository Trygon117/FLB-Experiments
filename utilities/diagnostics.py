import torch
import torch.nn as nn
import os
import json

def inspect_window(data):
    if data['window_idx'] == 0:
        raw_loss = data['raw_loss']
        slot_losses = raw_loss.mean(dim=0).tolist()
        formatted = [round(l, 2) for l in slot_losses]
        print(f"Slot losses: {formatted}")

        # 1. Decode the preceding context window that the model just read
        input_ids = data['input'][0].tolist()
        context_text = decode(input_ids)

        # 2. Decode the highest probability token for each future slot
        pred_ids = torch.argmax(data['predictions'][0], dim=-1).tolist()
        pred_text = decode(pred_ids)

        # 3. Decode the actual ground truth tokens that came next
        target_ids = data['targets'][0].tolist()
        target_text = decode(target_ids)

        # 4. Print the trailing part of context along with the predictions
        # Slicing the last 48 characters keeps the output easy to read in the terminal
        print(f"Context:        {repr(context_text)}")
        print(f"Pred:           {repr(pred_text)}")
        print(f"Target:         {repr(target_text)}")

# Streaming callback for real-time generation preview
def stream_character(data):
    char = decode(data['new_tokens'][0].tolist())
    
    # Swap raw line breaks with visible text markers
    safe_char = char.replace('\n', '\\n').replace('\r', '\\r')
    
    print(safe_char, end="", flush=True)

# Class to compare model gradients. (Class so it doesnt need to write to disk every step)
class DiagnosticTracker:
    def __init__(self, vocab, model_name="model"):
        self.vocab = vocab
        self.save_path = "training_artifacts/" + model_name + "_diagnostics.json"
        self.records = []

    def _decode(self, token_ids):
        if self.vocab is None:
            return str(token_ids)
        
        if isinstance(self.vocab, (list, str)):
            return "".join([self.vocab[i] if 0 <= i < len(self.vocab) else "?" for i in token_ids])
        
        if isinstance(self.vocab, dict):
            return "".join([self.vocab.get(i, "?") for i in token_ids])
        
        return str(token_ids)

    def __call__(self, data):
        window_idx = data.get('window_idx', 0)
        model = data.get('model', None)
        raw_loss = data['raw_loss']
        slot_losses = [round(float(val), 4) for val in raw_loss.mean(dim=0).tolist()]

        context_ids = data['input'][0].tolist()
        pred_ids = torch.argmax(data['predictions'][0], dim=-1).tolist()
        target_ids = data['targets'][0].tolist()

        context_text = self._decode(context_ids)
        pred_text = self._decode(pred_ids)
        target_text = self._decode(target_ids)

        print(f"Window       {window_idx}")
        print(f"Slot losses: {slot_losses}")
        print(f"Context:     {repr(context_text[-48:])}")
        print(f"Pred:        {repr(pred_text)}")
        print(f"Target:      {repr(target_text)} \n")

        param_grads = {}
        layer_group_grads = {}

        if model is not None:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    norm_val = round(float(param.grad.norm().item()), 6)
                    param_grads[name] = norm_val

                    for layer_idx in range(getattr(model, "num_layers", 4)):
                        if f"layers.{layer_idx}" in name or f"encoder.layers.{layer_idx}" in name:
                            key = f"layer_{layer_idx}"
                            current_sq = layer_group_grads.get(key, 0.0) ** 2
                            layer_group_grads[key] = round((current_sq + norm_val ** 2) ** 0.5, 6)

        self.records.append({
            'window_idx': window_idx,
            'step_loss': round(float(data['step_loss']), 4),
            'slot_losses': slot_losses,
            'context': context_text[-48:],
            'pred': pred_text,
            'target': target_text,
            'layer_summary': layer_group_grads,
            'param_gradients': param_grads
        })

    def save(self, custom_path=None):
        target_file = custom_path if custom_path is not None else self.save_path
        os.makedirs(os.path.dirname(target_file), exist_ok=True)
        with open(target_file, "w") as f:
            json.dump(self.records, f, indent=2)
        print(f"Saved diagnostic telemetry to {target_file}")