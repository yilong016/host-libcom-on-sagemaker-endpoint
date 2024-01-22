# This is the file that implements a flask server to do inferences. 
# It's the file that you will modify to implement the scoring for your own algorithm.

from __future__ import print_function
from naive_composition.generate_composite_image import *

# 存储图像的临时目录

background_path = "uploads/bg_000.png"
foreground_path = "uploads/fg_000.jpg"
foreground_mask_path = "uploads/fg_mask_000.png"
bbox=[376, 187, 550, 791]

print('start simple process...')
fg_img   = read_image_opencv(foreground_path)
fg_mask  = read_mask_opencv(foreground_mask_path)
bg_img   = read_image_opencv(background_path)
comp_img,comp_mask=simple_composite_image(bg_img, fg_img, fg_mask, bbox)

print(f'comp is:{comp_img}')
print(f'comp_mask is:{comp_mask}')
