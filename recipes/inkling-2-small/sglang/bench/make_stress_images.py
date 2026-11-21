"""Stress pictures for bench/image_stress.py: very large, odd formats, and broken inputs (Pillow; run in the image):
    docker run --rm --network none -v "$PWD/logs/test_images":/o -v "$PWD/bench":/b:ro \
      --entrypoint /opt/sglang/bin/python3 lmsysorg/sglang:dev-inkling-2-small /b/make_stress_images.py /o
MIT License, Copyright (c) 2026 Aiden Le.
"""
import io
import os
import sys
from PIL import Image, ImageDraw, ImageFont

o = sys.argv[1] if len(sys.argv) > 1 else "/o"


def font(size):
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default(size=size)


def card(size, text, bg="white", fg="black", mode="RGB", fsize=300):
    im = Image.new(mode, size, bg)
    d = ImageDraw.Draw(im)
    box = d.textbbox((0, 0), text, font=font(fsize))
    x, y = (size[0] - (box[2] - box[0])) // 2, (size[1] - (box[3] - box[1])) // 2
    d.text((x, y), text, fill=fg, font=font(fsize))
    return im


card((4096, 4096), "KIWI 88").save(f"{o}/huge.png")                              # 4096 x 4096: at the cap
card((6000, 4000), "PLUM 19", fsize=500).save(f"{o}/wide.jpg", quality=90)       # above the cap: downscaled
rgba = Image.new("RGBA", (900, 500), (0, 0, 0, 0))
ImageDraw.Draw(rgba).text((80, 150), "MANGO 5", fill=(200, 0, 0, 255), font=font(160))
rgba.save(f"{o}/rgba.png")                                                       # transparency
card((900, 500), "LIME 3", bg=255, fg=0, mode="L", fsize=160).save(f"{o}/gray.png")   # greyscale
card((900, 500), "PEAR 2", fsize=160).save(f"{o}/pear.webp")                     # WebP
Image.new("RGB", (1, 1), (0, 128, 0)).save(f"{o}/one_px.png")                    # 1 x 1
buf = io.BytesIO(); card((600, 400), "HALF", fsize=120).save(buf, "PNG")
open(f"{o}/truncated.png", "wb").write(buf.getvalue()[: len(buf.getvalue()) // 2])  # broken PNG
open(f"{o}/notimage.txt", "wb").write(b"this is not a picture\n" * 20)            # not a picture
print("made", sorted(os.listdir(o)))
