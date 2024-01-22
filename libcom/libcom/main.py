from fopa_heat_map.fopa_heat_map import FOPAHeatMapModel
from image_harmonization.image_harmonization import ImageHarmonizationModel
from naive_composition.generate_composite_image import *
import cv2
import os
import base64
import random
from datetime import datetime
from PIL import Image
import numpy as np
import io


# workd dir
upload_dir = "uploads/"
result_dir = "results/"

def save_image_with_datetime(image, save_dir):
    # generate timestamps
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # generate a random str
    random_number = str(random.randint(1, 999)).zfill(3)

    # construct file name
    file_name = f"harmony_{current_datetime}_{random_number}.png"

    # consutruct file path
    save_path = os.path.join(save_dir, file_name)

    # save files
    cv2.imwrite(save_path, image)

    # return file path
    return save_path

def save_image_from_base64(base64_data, output_folder, prefix):
    try:
        decoded_image = base64.b64decode(base64_data)
        os.makedirs(output_folder, exist_ok=True)        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_number = f"{random.randint(1, 1000):03d}"
        file_name = f"{prefix}_{timestamp}_{random_number}.png"
        file_path = os.path.join(output_folder, file_name)        
        with open(file_path, 'wb') as file:
            file.write(decoded_image)
        return file_path
    except Exception as e:
        print(f"Error saving image: {e}")
        return None

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_image


def read_image_opencv(input):
    if isinstance(input, str):
        assert os.path.exists(input), input
        input = cv2.imread(input, cv2.IMREAD_COLOR)
    elif isinstance(input, Image.Image):
        input = pil_to_opencv(input)
    return input

def fopa_algorithm(foreground_path, foreground_mask_path, background_path):
    # process fopa
    net = FOPAHeatMapModel(device=0)

    bboxes, heatmaps = net(foreground_path, foreground_mask_path, background_path, 
            cache_dir=os.path.join(result_dir, 'cache'), 
            heatmap_dir=os.path.join(result_dir, 'heatmap'))

    return bboxes, heatmaps[0]

def getbbox(img_mask_path):
    # get bbox from foreground mask, used to get foreground width and height
    mask=cv2.imread(img_mask_path,cv2.IMREAD_GRAYSCALE)
    x_indices,y_indices = cv2.findNonZero(mask).T
    print(x_indices)
    print(min(x_indices[0]),max(x_indices[0]))
    x,y,w,h = min(x_indices[0]),min(y_indices[0]),max(x_indices[0])-min(x_indices[0]),max(y_indices[0])-min(y_indices[0])
    return x,y,w,h

def find_corresponding_coordinate(heatmap_path, background_path):
    # find highest spot in heatmap
    block_size = 50  # block size 
    # read heatmap as A
    image_A = Image.open(heatmap_path).convert('L')  # convert to gray
    width_A, height_A = image_A.size

    # calculate blocks
    blocks_x = width_A // block_size
    blocks_y = height_A // block_size

    # find highlight block
    brightest_block = (0, 0)
    brightest_avg_value = 0
    for x in range(blocks_x):
        for y in range(blocks_y):
            block_values = []
            for i in range(block_size):
                for j in range(block_size):
                    pixel_x = x * block_size + i
                    pixel_y = y * block_size + j
                    if pixel_x < width_A and pixel_y < height_A:
                        pixel_value = image_A.getpixel((pixel_x, pixel_y))
                        block_values.append(pixel_value)
            avg_value = sum(block_values) / len(block_values)
            if avg_value > brightest_avg_value:
                brightest_avg_value = avg_value
                brightest_block = (x, y)

    # find highlight spot in highlist block
    brightest_coordinate = (brightest_block[0] * block_size, brightest_block[1] * block_size)
    brightest_value = 0
    for i in range(block_size):
        for j in range(block_size):
            pixel_x = brightest_coordinate[0] + i
            pixel_y = brightest_coordinate[1] + j
            if pixel_x < width_A and pixel_y < height_A:
                pixel_value = image_A.getpixel((pixel_x, pixel_y))
                if pixel_value > brightest_value:
                    brightest_value = pixel_value
                    brightest_coordinate = (pixel_x, pixel_y)

    # read background images as B
    image_B = Image.open(background_path)
    width_B, height_B = image_B.size

    # find the target dot coordinates in B as highlight spot in A
    scale_x = width_B / width_A
    scale_y = height_B / height_A
    high_spot_x = int(brightest_coordinate[0] * scale_x)
    high_spot_y = int(brightest_coordinate[1] * scale_y)
    
    return high_spot_x, high_spot_y

def image_compose(bg_img_path, fg_img_path, fg_mask_path, bbox):
    # simple compose images
    fg_img   = read_image_opencv(fg_img_path)
    fg_mask  = read_mask_opencv(fg_mask_path)
    bg_img   = read_image_opencv(bg_img_path)
    comp_img,comp_mask=simple_composite_image(bg_img, fg_img, fg_mask, bbox)
    print(f'comp_img is: {comp_img}')
    print(f'comp_mask is: {comp_mask}')
    return comp_img, comp_mask

def pct_mode(comp_img_path,comp_mask_path):
    # process pct harmonization
    comp_img = read_image_opencv(comp_img_path)
    comp_mask= read_mask_opencv(comp_mask_path)
    pct_model = ImageHarmonizationModel(model_type='PCTNet')
    pct_res   = pct_model(comp_img, comp_mask)
    print(f'pct_res is: {pct_res}')
    return pct_res
    
def cdt_mode(comp_img_path,comp_mask_path):
    # process cdt harmonization
    comp_img = read_image_opencv(comp_img_path)
    comp_mask= read_mask_opencv(comp_mask_path)
    cdt_model = ImageHarmonizationModel(model_type='CDTNet')
    cdt_res   = cdt_model(comp_img, comp_mask)
    print(f'cdt_res is: {cdt_res}')
    return cdt_res


from shadow_generation import ShadowGenerationModel
def shadow_algorithm(comp_image_path, comp_mask_path, number):
    # generate shadow
    net = ShadowGenerationModel(device=0, model_type='ShadowGeneration')
    preds = net(comp_image_path, comp_mask_path, number=number)
    return preds[0]

