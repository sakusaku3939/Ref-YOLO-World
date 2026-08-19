import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def load_sd(path):
    ckpt = torch.load(path, map_location="cpu")
    return ckpt, ckpt.get("state_dict", ckpt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["copy_matching", "open_gate"], default="open_gate")
    ap.add_argument("--gate-bias", type=float, default=8.0)
    args = ap.parse_args()

    src_ckpt, src = load_sd(args.source)
    tmpl_ckpt, tmpl = load_sd(args.template)
    target = OrderedDict((k, v.detach().cpu().clone()) for k, v in tmpl.items())

    copied, skipped_shape, missing_in_src = [], [], []
    for k, tv in list(target.items()):
        sv = src.get(k)
        if sv is None:
            missing_in_src.append(k)
            continue
        if tuple(sv.shape) == tuple(tv.shape):
            target[k] = sv.detach().cpu().clone()
            copied.append(k)
        else:
            skipped_shape.append((k, tuple(sv.shape), tuple(tv.shape)))

    initialized = []
    for k, v in list(target.items()):
        if k not in src:
            if ".guide_value_fc." in k or ".context_proj." in k or k.endswith(".scale_context") or ".out_proj." in k:
                target[k] = torch.zeros_like(v)
                initialized.append((k, "zero"))
            elif ".pe_proj.2." in k:
                target[k] = torch.zeros_like(v)
                initialized.append((k, "zero"))

    if args.mode == "open_gate":
        for k, v in list(target.items()):
            if k.endswith("attn_block.guide_fc.weight") or k.endswith("attn_block.guide_fc.bias"):
                target[k] = torch.zeros_like(v)
                initialized.append((k, "zero_for_constant_gate"))
            elif k.endswith("attn_block.bias"):
                target[k] = torch.full_like(v, args.gate_bias)
                initialized.append((k, f"gate_bias_{args.gate_bias}"))

    out = dict(tmpl_ckpt)
    out["state_dict"] = target
    out["meta"] = dict(out.get("meta", {}))
    out["meta"].update({
        "conversion": "yolo_world_to_ref_yolo_world_template",
        "source_checkpoint": args.source,
        "template_checkpoint": args.template,
        "mode": args.mode,
        "gate_bias": args.gate_bias,
        "copied_matching_keys": len(copied),
        "missing_in_source_keys": len(missing_in_src),
        "skipped_shape_keys": len(skipped_shape),
        "initialized_keys": len(initialized),
    })
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.output)

    print("saved", args.output)
    print("mode", args.mode)
    print("copied_matching_keys", len(copied))
    print("missing_in_source_keys", len(missing_in_src))
    print("skipped_shape_keys", len(skipped_shape))
    print("initialized_keys", len(initialized))
    print("sample_missing", missing_in_src[:30])
    print("sample_initialized", initialized[:30])


if __name__ == "__main__":
    main()
