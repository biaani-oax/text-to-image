#!/usr/bin/env python3
"""Genera imágenes con SD 1.5 + LCM-LoRA (CPU, sin internet)."""
import argparse
import os
import random
import re
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"

BASE = Path(__file__).parent

NEG_COMUN = ("blurry, lowres, deformed, bad anatomy, extra limbs, "
             "text, watermark, signature, jpeg artifacts")

ESTILOS = {
    "caricatura": ("children's book illustration, cartoon, thick clean outlines, big expressive eyes, soft vibrant colors, whimsical, highly detailed",
                   "photo, realistic"),
    "lapiz": ("detailed graphite pencil drawing, crosshatching, fine shading, sketchbook paper texture, monochrome, intricate details",
              "photo, realistic, color"),
    "acuarela": ("watercolor painting, soft washes, bleeding colors, visible paper texture, loose brushstrokes, pastel palette",
                 "photo, realistic"),
    "tinta": ("black ink illustration, vintage engraving style, fine hatching, high contrast, old book illustration",
              "photo, realistic, color"),
    "isometrico": ("isometric miniature diorama, soft pastel colors, warm lighting, clean render, highly detailed",
                   "photo, realistic"),
    "vector": ("flat vector illustration, vector art, clean bold lines, solid colors, limited color palette, simple geometric shapes, no gradients, sticker design, plain white background",
               "photo, realistic, gradient, shadow, texture, noise, 3d render, sketch"),
    "crochet": ("crocheted amigurumi toy, made of yarn, knitted wool texture, visible stitches, soft and fuzzy, felt details, cute, handmade craft, soft studio lighting",
                "drawing, painting, smooth, plastic, flat colors"),
    "anime": ("anime style illustration, cel shading, clean lineart, vibrant colors, big expressive eyes, detailed background",
              "photo, realistic, 3d render"),
    "pixel": ("pixel art, 16-bit retro video game sprite, limited color palette, sharp square pixels, dithering, simple background",
              "photo, realistic, smooth, blurry, antialiasing"),
    "comic": ("pop art comic book illustration, bold black outlines, halftone dots, vivid primary colors, dynamic composition, retro 1960s style",
              "photo, realistic, 3d render, muted colors"),
    "arcilla": ("claymation style, cute plasticine clay figures, handmade stop motion, soft studio lighting, fingerprint texture, vibrant colors",
                "drawing, flat, vector"),
    "synthwave": ("1980s synthwave retrowave illustration, glowing neon lines on a dark digital grid, magenta and cyan neon glow, purple sunset horizon, chrome highlights, highly detailed",
                  "photo, realistic, daylight, muted colors"),
    "arcade": ("1980s arcade video game art, 16-bit pixel art, vibrant saturated colors, neon glow, CRT scanlines, retro game screen, detailed sprites",
               "photo, realistic, smooth, blurry, muted colors"),
}
def parse_args():
    p = argparse.ArgumentParser(description="Texto a imagen con SD 1.5 + LCM-LoRA")
    p.add_argument("prompt", help="qué dibujar (en inglés funciona mejor)")
    p.add_argument("-e", "--estilo", choices=list(ESTILOS), default="caricatura")
    p.add_argument("-p", "--pasos", type=int, choices=range(1, 20),
                   default=4, metavar="1-10", help="pasos de difusión (default 4)")
    p.add_argument("-s", "--semilla", type=int, default=None,
                   help="semilla para repetir un resultado (default: aleatoria)")
    p.add_argument("-o", "--salida", default="salidas", help="carpeta de salida")
    p.add_argument("--svg", action="store_true", help="intenta vectorizar con vtracer")
    return p.parse_args()


def a_svg(png: Path, estilo: str):
    try:
        import vtracer
    except ImportError:
        print("Falta vtracer: pip install vtracer "
              "(si falla en tu versión de Python, usa un venv con 3.12)")
        return
    svg = png.with_suffix(".svg")
    vtracer.convert_image_to_svg_py(
        str(png), str(svg),
        colormode="binary" if estilo in ("lapiz", "tinta") else "color",
        filter_speckle=8,
        color_precision=5,
    )
    print("SVG:", svg)


def main():
    args = parse_args()

    import torch
    from diffusers import AutoPipelineForText2Image, LCMScheduler

    pipe = AutoPipelineForText2Image.from_pretrained(
        str(BASE / "models" / "sd15"), torch_dtype=torch.float32, variant="fp16",
        safety_checker=None, requires_safety_checker=False,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(
        str(BASE / "models" / "loras" / "lcm"),
        weight_name="pytorch_lora_weights.safetensors", adapter_name="lcm",
    )
    pipe.set_adapters(["lcm"], adapter_weights=[1.0])

    seed = args.semilla if args.semilla is not None else random.randint(0, 2**31 - 1)
    negativo = f"{NEG_COMUN}, {ESTILOS[args.estilo][1]}"

    img = pipe(
        f"{args.prompt}, {ESTILOS[args.estilo][0]}",
        negative_prompt=negativo,
        num_inference_steps=args.pasos,
        guidance_scale=2.0,
        height=512, width=512,
        generator=torch.Generator().manual_seed(seed),
    ).images[0]

    salida = BASE / args.salida
    salida.mkdir(exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", args.prompt.lower()).strip("-")[:40]
    png = salida / f"{slug}_{args.estilo}_p{args.pasos}_s{seed}.png"
    img.save(png)
    print(f"Guardada: {png}  (semilla {seed})")

    if args.svg:
        a_svg(png, args.estilo)


if __name__ == "__main__":
    main()