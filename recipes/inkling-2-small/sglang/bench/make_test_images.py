"""The five test pictures of bench/image_check.py, with known content: a blue circle on red, three green squares and a
yellow triangle, the text "ORANGE 7351", a 3000 x 2000 picture with "ZEBRA 64" in a black box, a 40 x 40 purple square.
Needs Pillow, which the serving image has:
    docker run --rm --network none -v "$PWD/logs/test_images":/o -v "$PWD/bench":/b:ro \
      --entrypoint /opt/sglang/bin/python3 lmsysorg/sglang:dev-inkling-2-small /b/make_test_images.py /o
MIT License, Copyright (c) 2026 Aiden Le.
"""
from PIL import Image, ImageDraw, ImageFont
import os
import sys
o=sys.argv[1] if len(sys.argv) > 1 else '/o'
def font(size):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf','/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'):
        if os.path.exists(p): return ImageFont.truetype(p,size)
    try: return ImageFont.load_default(size=size)
    except Exception: return ImageFont.load_default()
# 1: blue circle on red background, small
im=Image.new('RGB',(512,512),(220,30,30)); d=ImageDraw.Draw(im); d.ellipse((128,128,384,384),fill=(30,60,220)); im.save(f'{o}/circle.png')
# 2: three green squares and one yellow triangle on white
im=Image.new('RGB',(800,600),'white'); d=ImageDraw.Draw(im)
for x in (60,260,460): d.rectangle((x,80,x+150,230),fill=(20,160,40))
d.polygon([(330,520),(430,320),(530,520)],fill=(240,210,0)); im.save(f'{o}/shapes.png')
# 3: big text
im=Image.new('RGB',(1400,512),'white'); d=ImageDraw.Draw(im); d.text((60,150),'ORANGE 7351',fill='black',font=font(140))
assert d.textbbox((60,150),'ORANGE 7351',font=font(140))[2] < 1400 - 40  # the whole text inside the picture
im.save(f'{o}/text.png')
# 4: large image 3000x2000 with text in a corner
im=Image.new('RGB',(3000,2000),(235,235,235)); d=ImageDraw.Draw(im); d.rectangle((2100,1500,2950,1950),fill=(0,0,0)); d.text((2150,1600),'ZEBRA 64',fill='white',font=font(160)); im.save(f'{o}/large.png')
# 5: tiny image 40x40 purple
Image.new('RGB',(40,40),(128,0,160)).save(f'{o}/tiny.png')
print('made', sorted(os.listdir(o)))
