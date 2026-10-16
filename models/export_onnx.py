#!/usr/bin/env python3
"""Export a YOLOv8 detector to ONNX for this project.

ONNX is the GPU-agnostic middle step. The .onnx file is portable and belongs in
the repo; the TensorRT .engine built from it is specific to one GPU, one
TensorRT version and one driver, and is gitignored on purpose.

    python3 export_onnx.py                      # yolov8n at 640x640
    python3 export_onnx.py --model yolov8s.pt   # a bigger model
    python3 export_onnx.py --dynamic-batch      # lets --batch > 1 work later

Then build the engine for the machine you are sitting at:

    ./build_engine.sh model.onnx model.engine

Why YOLOv8n by default: it is the smallest model in the family, and the
pipeline's job is to produce a steering command 20-30 times a second, not to
win a detection benchmark. A larger model mostly buys recall on small distant
objects, which is not the bottleneck when the target is a person or a ball a
couple of metres away. Start with n; you can measure the difference with
--bench later and decide for yourself.
"""

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Export a YOLOv8 model to ONNX for reflex-cv")
    ap.add_argument("--model", default="yolov8n.pt",
                    help="ultralytics model name or .pt path (default yolov8n.pt)")
    ap.add_argument("--imgsz", type=int, default=640,
                    help="square input size (default 640)")
    ap.add_argument("--opset", type=int, default=12,
                    help="ONNX opset (default 12; raise it only if the export "
                         "complains about an unsupported op)")
    ap.add_argument("--out", default="model.onnx",
                    help="output path (default model.onnx)")
    ap.add_argument("--dynamic-batch", action="store_true",
                    help="export with a dynamic batch dimension, so the engine "
                         "can be built for batch > 1 later")
    ap.add_argument("--no-simplify", action="store_true",
                    help="skip onnx-simplifier (try this if the export fails "
                         "during simplification)")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "error: the 'ultralytics' package is not installed.\n"
            "\n"
            "    pip install ultralytics onnx onnxsim\n"
            "\n"
            "This is only needed ONCE, to produce the .onnx file, and only on\n"
            "whichever machine does the export - the resulting .onnx is\n"
            "committed to the repo and the C++ pipeline never imports Python.",
            file=sys.stderr)
        return 1

    print(f"loading {args.model}")
    model = YOLO(args.model)

    print(f"exporting to ONNX at {args.imgsz}x{args.imgsz}, opset {args.opset}"
          f"{', dynamic batch' if args.dynamic_batch else ''}")
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=not args.no_simplify,
        # `dynamic` makes the batch axis symbolic. Without it the engine is
        # locked to batch 1 and main.cpp's --batch flag cannot do anything.
        dynamic=args.dynamic_batch,
        # NMS stays OUT of the graph on purpose: src/postprocess.cpp does the
        # decoding and NMS on the CPU, where it is a few hundred microseconds,
        # unit-testable, and easy to change. Baking NMS into the ONNX graph
        # makes the TensorRT conversion fussier and the output shape
        # export-dependent, for no measurable gain at this scale.
        nms=False,
    )

    if path and os.path.abspath(str(path)) != os.path.abspath(args.out):
        os.replace(str(path), args.out)
    out = args.out

    size_mb = os.path.getsize(out) / (1024 * 1024)
    print(f"\nwrote {out} ({size_mb:.1f} MB)")

    # Report the real I/O shapes - this is what main.cpp will read off the
    # engine, and it is worth seeing now rather than being surprised later.
    try:
        import onnx
        m = onnx.load(out)
        print("\ngraph I/O:")
        for group, items in (("input", m.graph.input),
                             ("output", m.graph.output)):
            for t in items:
                dims = [
                    (d.dim_param or str(d.dim_value))
                    for d in t.type.tensor_type.shape.dim
                ]
                print(f"  {group:6s} {t.name}  [{' x '.join(dims)}]")
        print("\nAn output of [1 x 84 x 8400] is the usual YOLOv8 COCO shape:")
        print("  84 = 4 box values + 80 class scores, 8400 = candidate boxes.")
        print("  src/main.cpp infers this layout from the engine, so a model")
        print("  with a different class count needs no code change.")
    except ImportError:
        print("(install onnx to print the graph's input/output shapes)")

    print("\nnext:")
    print(f"  ./build_engine.sh {out} model.engine")
    print("  ...on EACH machine. Engines are not portable between GPUs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
