"""Opt-in activation capture and local Linear precision comparison.

Forward-output error is a calibration signal, not evidence of scientific video
accuracy. Captures are bounded to the first selected inputs per expert/layer.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path

import torch
import torch.nn.functional as F


def install_capture(model, directory, expert, patterns, max_tokens=32, weight_identity=''):
    if not directory or not patterns:
        return
    if max_tokens < 1:
        raise ValueError('Capture token limit must be positive')
    from safetensors.torch import save_file
    root = Path(directory) / expert
    root.mkdir(parents=True, exist_ok=True)
    for name, module in model.named_modules():
        if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
            continue
        if not hasattr(module, 'in_features'):
            continue
        path = root / (name+'.safetensors')
        if path.exists():
            continue
        def capture(current, inputs, *, destination=path, layer=name):
            if destination.exists() or not inputs or not torch.is_tensor(inputs[0]):
                return
            value = inputs[0].detach().reshape(-1, current.in_features)[:max_tokens].cpu().contiguous()
            save_file({'input': value}, str(destination), metadata={'layer': layer, 'expert': expert,
                'weight_identity': weight_identity,
                'scope': 'first encountered input tokens; not all timesteps or experiments'})
        module.register_forward_pre_hook(capture)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures',required=True)
    parser.add_argument('--checkpoint-file',required=True)
    parser.add_argument('--lora-file')
    parser.add_argument('--expert',choices=['high_noise_model','low_noise_model'],required=True)
    parser.add_argument('--quantization',choices=['per_tensor','per_channel'],default='per_tensor')
    parser.add_argument('--relative-error-limit',type=float,default=.01)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import math
    if not math.isfinite(args.relative_error_limit) or args.relative_error_limit <= 0:
        raise ValueError('relative-error-limit must be positive')
    from contextlib import ExitStack
    from safetensors import safe_open
    from kaggle_single_expert import _checkpoint_key_map, _dequantize_fp8_tensor
    from quantization.optimization.lora import plan_lora_tensors, merge_lora_weight, _module_path_from_lora_key
    from runtime_cache import file_identity, digest
    checkpoint_identity=file_identity(args.checkpoint_file)
    adapter_identity=file_identity(args.lora_file) if args.lora_file else None
    expected_weight_identity=digest({'source':checkpoint_identity, 'adapter':adapter_identity})
    records=[]
    paths=sorted(Path(args.captures).glob('*.safetensors'))
    if not paths:
        raise ValueError('No captured layer inputs found')
    with ExitStack() as stack:
        source=stack.enter_context(safe_open(args.checkpoint_file,framework='pt',device='cpu'))
        keymap=_checkpoint_key_map(list(source.keys()))
        adapter=stack.enter_context(safe_open(args.lora_file,framework='pt',device='cpu')) if args.lora_file else None
        for path in paths:
            with safe_open(str(path),framework='pt',device='cpu') as handle:
                metadata=handle.metadata() or {}
                if metadata.get('expert')!=args.expert:
                    raise ValueError('Capture/checkpoint expert mismatch')
                if metadata.get('weight_identity')!=expected_weight_identity:
                    raise ValueError('Captured inputs belong to different base weights or adapter')
                name=metadata['layer']
                inputs=handle.get_tensor('input').float()
            weight_name=name+'.weight'
            raw=source.get_tensor(keymap[weight_name])
            scale_key=next((keymap[k] for k in (name+'.scale_weight',name+'.weight_scale') if k in keymap),None)
            weight=_dequantize_fp8_tensor(raw,source.get_tensor(scale_key) if scale_key else None)
            if adapter is not None and name.startswith('blocks.'):
                all_keys=set(adapter.keys())
                down_key=next((key for key in all_keys if key.endswith('.lora_down.weight')
                    and _module_path_from_lora_key(key)==name),None)
                if down_key is None:
                    raise ValueError(f'Adapter has no matching target for {name}')
                stem=down_key[:-len('.lora_down.weight')]
                keys={stem+suffix for suffix in ('.lora_down.weight','.lora_up.weight','.alpha')}
                if not keys.issubset(all_keys):
                    raise ValueError(f'Incomplete adapter factors for {name}')
                plans=plan_lora_tensors({name:(weight.shape[1],weight.shape[0])},keys,
                    {key:tuple(adapter.get_slice(key).get_shape()) for key in keys},expected_targets={name})
                plan=plans[name]
                weight=merge_lora_weight(weight,adapter.get_tensor(plan.down_key),
                    adapter.get_tensor(plan.up_key),adapter.get_tensor(plan.alpha_key))
            reference=F.linear(inputs,weight.half().float())
            maximum=weight.abs().amax(dim=1,keepdim=True) if args.quantization=='per_channel' else weight.abs().max()
            scale=torch.where(maximum>0,maximum/127,torch.ones_like(maximum))
            quantized=torch.round(weight/scale).clamp(-127,127).to(torch.int8)
            reconstructed=quantized.half().mul_(scale).float()
            candidate=F.linear(inputs,reconstructed)
            relative=float(torch.linalg.vector_norm(candidate-reference)/torch.linalg.vector_norm(reference).clamp_min(1e-12))
            records.append({'layer':name,'capture_sha256':file_identity(path)['sha256'],
                'relative_l2_error':relative,'keep_fp16_candidate':relative>args.relative_error_limit})
    output={'checkpoint':checkpoint_identity,
        'adapter':adapter_identity,
        'expert':args.expert,'quantization':args.quantization,
        'relative_error_limit':args.relative_error_limit,'layers':records,
        'keep_fp16':[r['layer'] for r in records if r['keep_fp16_candidate']],
        'limits':'CPU float32 Linear arithmetic on FP16-rounded weights and captured quantized-path inputs. Bias cancels in this comparison. This does not measure end-to-end video quality or prove optimal sensitivity weights.'}
    Path(args.output).write_text(json.dumps(output,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
