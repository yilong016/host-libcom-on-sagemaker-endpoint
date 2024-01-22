from fopa_heat_map.fopa_heat_map import FOPAHeatMapModel
import os

# 存储图像的临时目录
upload_dir = "uploads/"
result_dir = "results/"

background_path = "uploads/bg_000.png"
foreground_path = "uploads/fg_000.jpg"
foreground_mask_path = "uploads/fg_mask_000.png"


net = FOPAHeatMapModel(device=0)

bboxes, heatmaps = net(foreground_path, foreground_mask_path, background_path, 
            cache_dir=os.path.join(result_dir, 'cache'), 
            heatmap_dir=os.path.join(result_dir, 'heatmap'))

print(bboxes)
print(heatmaps[0])
