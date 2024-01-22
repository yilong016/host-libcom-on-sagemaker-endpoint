# This is the file that implements a flask server to do inferences. 
# It's the file that you will modify to implement the scoring for your own algorithm.

from __future__ import print_function

from main import *
from utils import *

# 存储图像的临时目录

result_dir='results/'

comp_image_path = "uploads/m1_composition_000.png"
comp_mask_path = "uploads/m1_composition_mask_000.png"

print('start shadow_result process...')
shadow_result = shadow_algorithm(comp_image_path, comp_mask_path, 1)
print(f'shadow_result is :{shadow_result}')

# 保存文件
shadow_result_path = save_image_with_datetime(shadow_result, result_dir)
# 将文件转为base64
result_image = encode_image(shadow_result_path)


