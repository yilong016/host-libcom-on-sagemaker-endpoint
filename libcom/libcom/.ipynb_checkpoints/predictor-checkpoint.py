# This is the file that implements a flask server to do inferences. 
# It's the file that you will modify to implement the scoring for your own algorithm.

from __future__ import print_function

from main import *
import os
import json
import flask


# work dir
upload_dir = "uploads/"
result_dir = "results/"

# temp for heatmap
cache_dir = os.path.join(result_dir, 'cache')
heatmap_dir = os.path.join(result_dir, 'heatmap')
# ...

prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
output_path = os.path.join(prefix, 'output')
saved_param_path = os.path.join(model_path, 'hyperparameters.json')


# The flask app for serving predictions
app = flask.Flask(__name__)

# The predictor is designed to load data for prediction either from a specified S3 location or from serialized CSV

@app.route('/ping', methods=['GET'])
def ping():
    #Determine if the container is working and healthy. In this sample container
    # Declare it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None 
    status = 200 #if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    #Do an inference on a single batch of data. 
    # Accept either a n S3 location containing a CSV file or from a serialized string, 
    # convert it to a pandas data frame for internal use and 
    # convert the predictions back to CSV        
    data = None
    status = None
    print("come into invocations")

    # Load CSV to DataFrame
    #if flask.request.content_type != 'text/csv':
    #    return flask.Response(response='This predictor only supports CSV data', status=415, mimetype='text/plain')
    data = json.loads(flask.request.data)
    function_name = data.get("function",None)
    if function_name == "heatmap_compose":
        result = heatmap_compose(data)
    elif function_name == "simple_compose":
        result = simple_compose(data)
    elif function_name == "pct_mode":
        result = pctnet(data)
    elif function_name == "cdt_mode":
        result = cdtnet(data)
    elif function_name == "gn_shadow":
        result = gn_shadow(data)
    else:
        return flask.Response(response='not support this functions\n', status=status, mimetype='application/json')    
    return result

def heatmap_compose(data):
    #get images base64 encoding from input
    bg_b64 = data.get("background",None)
    fg_b64 = data.get("foreground",None)
    fg_mask_b64 = data.get("foreground_mask",None)
    scale_n = data.get("scale_x",1)

    #save files
    background_path = save_image_from_base64(bg_b64, upload_dir, "background")
    print("background path is:",background_path)
    foreground_path = save_image_from_base64(fg_b64, upload_dir, "foreground")
    print("foreground path is:",foreground_path)
    foreground_mask_path = save_image_from_base64(fg_mask_b64, upload_dir, "foreground_mask")
    print("foreground mask path is:",foreground_mask_path)

    print("begin fopa process")

    # get fopa heatmap
    bboxes, heatmaps_path = fopa_algorithm(foreground_path, foreground_mask_path, background_path)
    print('bboxes and heatmaps are: ')
    print(bboxes, heatmaps_path)
    #"results/heatmap/foreground_20240115_165614_826_background_20240115_165614_049_33_44_0.05555555555555555.jpg"
    heatmaps=encode_image(heatmaps_path)
    
    # obtain foreground object x,y,w,h
    x,y,w,h=getbbox(foreground_mask_path)
    high_x, high_y = find_corresponding_coordinate(heatmaps_path, background_path)
    
    # process scale of foreground object 
    w = int(w*scale_n)
    h = int(h*scale_n)
    
    # get bbox to make images composition
    bbox = [high_x-int(0.5*w),high_y-h,high_x+int(0.5*w),high_y]

    comp_img, comp_mask = image_compose(background_path,foreground_path,foreground_mask_path,bbox)
    
    #save files
    comp_img_path=save_image_with_datetime(comp_img,upload_dir)
    comp_mask_path=save_image_with_datetime(comp_mask,upload_dir)
    #convert to base64
    comp_image=encode_image(comp_img_path)
    comp_mask=encode_image(comp_mask_path)

    # return all results
    response = {
        "bboxes": bboxes,
        "heatmap_image": heatmaps,
        "comp_image":comp_image,
        "comp_mask":comp_mask
        }
    response = json.dumps(response)
    return response

def simple_compose(data):
    #get images base64 encoding from input
    bg_b64 = data.get("background",None)
    fg_b64 = data.get("foreground",None)
    fg_mask_b64 = data.get("foreground_mask",None)
    bbox=data.get("bbox",None)

    #save files
    bg_path=save_image_from_base64(bg_b64, upload_dir, "background")
    print("background path is:",bg_path)
    fg_path=save_image_from_base64(fg_b64, upload_dir, "foreground")
    print("foreground path is:",fg_path)
    fg_mask_path=save_image_from_base64(fg_mask_b64, upload_dir, "foreground_mask")
    print("foreground mask path is:",fg_mask_path)

    #compose images
    comp_img, comp_mask = image_compose(bg_path,fg_path,fg_mask_path,bbox)
    
    #save files
    comp_img_path=save_image_with_datetime(comp_img,upload_dir)
    comp_mask_path=save_image_with_datetime(comp_mask,upload_dir)
    
    #convert to base64
    comp_image=encode_image(comp_img_path)
    comp_mask=encode_image(comp_mask_path)

    #return all results
    response = {
        "comp_image": comp_image,
        "comp_mask": comp_mask
        }
    response = json.dumps(response)
    return response

def pctnet(data):
    # get images base64 encoding from input
    comp_image_b64 = data.get("comp_image",None)
    comp_mask_b64 = data.get("comp_mask",None)

    # save files
    comp_image_path = save_image_from_base64(comp_image_b64, upload_dir, "comp_image")
    print("comp_image path is:",comp_image_path)
    comp_mask_path = save_image_from_base64(comp_mask_b64, upload_dir, "comp_mask")
    print("comp_mask path is:",comp_mask_path)
    
    # get pct mode harmonization
    pct_res=pct_mode(comp_image_path,comp_mask_path)
    #print(f'pct_res is: {pct_res}')

    # save files
    pct_res_path=save_image_with_datetime(pct_res,upload_dir)
    print("pct_res_path is:",pct_res_path)
    
    # convert to base64
    pct_result=encode_image(pct_res_path)
    #print("pct_result is:",pct_result)

    response = {
        "pct_result": pct_result
        }
    response = json.dumps(response)
    return response

def cdtnet(data):
    # get images base64 encoding from input
    comp_image_b64 = data.get("comp_image",None)
    comp_mask_b64 = data.get("comp_mask",None)
    
    # save files
    comp_image_path = save_image_from_base64(comp_image_b64, upload_dir, "comp_image")
    print("comp_image path is:",comp_image_path)
    comp_mask_path = save_image_from_base64(comp_mask_b64, upload_dir, "comp_mask")
    print("comp_mask path is:",comp_mask_path)

    # generate cdt mode harmonization
    cdt_res=cdt_mode(comp_image_path,comp_mask_path)
    #print(f'pct_res is: {cdt_res}')

    # save files
    cdt_res_path=save_image_with_datetime(cdt_res,upload_dir)
    print("cdt_res_path is:",cdt_res_path)
    
    # convert to base64
    cdt_result=encode_image(cdt_res_path)
    #print("cdt_result is:",cdt_result)
    
    response = {
        "cdt_result": cdt_result
        }
    response = json.dumps(response)
    return response


def gn_shadow(data):
    # get images base64 encoding from input
    comp_b64 = data.get("comp_image",None)
    comp_mask_b64 = data.get("comp_mask",None)
    number = data.get("number",1)
    
    # save files
    comp_image_path = save_image_from_base64(comp_b64, upload_dir, "comp_image")
    #print("comp_image is:",comp_image_path)
    comp_mask_path = save_image_from_base64(comp_mask_b64, upload_dir, "comp_mask")
    #print("comp_mask is:",comp_mask_path)
    #print(f'number is: {number}')

    # generate shadow
    shadow_result = shadow_algorithm(comp_image_path, comp_mask_path, number)
    #print(f'shadow_result is :{shadow_result}')
    
    # save files
    shadow_result_path = save_image_with_datetime(shadow_result, result_dir)

    # convert to base64
    result_image = encode_image(shadow_result_path)
    #print(f'result image is: {result_image}')
    response = {
        "shadow": result_image
        } 
    response = json.dumps(response)
    return response

