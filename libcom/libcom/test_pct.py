# This is the file that implements a flask server to do inferences. 
# It's the file that you will modify to implement the scoring for your own algorithm.

from __future__ import print_function

from main import *
from utils import *

# 存储图像的临时目录

comp_image_path = "uploads/m1_composition_000.png"
comp_mask_path = "uploads/m1_composition_mask_000.png"

comp_image   = read_image_opencv(comp_image_path)
comp_mask  = read_mask_opencv(comp_mask_path)

print('start pctnet process...')
pct_result = pct_mode(comp_image, comp_mask)
print(f'pct_result is :{pct_result}')


