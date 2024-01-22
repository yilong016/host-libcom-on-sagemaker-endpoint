#!/bin/bash

echo "models downloading start"
# fopa models
wget -O libcom/libcom/fopa_heat_map/pretrained_models/FOPA.pth "https://huggingface.co/BCMIZB/Libcom_pretrained_models/resolve/main/FOPA.pth"
wget -O libcom/libcom/fopa_heat_map/pretrained_models/SOPA.pth "https://huggingface.co/BCMIZB/Libcom_pretrained_models/resolve/main/SOPA.pth"


#image_harmonization model
wget -O libcom/libcom/image_harmonization/pretrained_models/CDTNet.pth "https://huggingface.co/BCMIZB/Libcom_pretrained_models/resolve/main/CDTNet.pth"
wget -O libcom/libcom/image_harmonization/pretrained_models/PCTNet.pth "https://huggingface.co/BCMIZB/Libcom_pretrained_models/resolve/main/PCTNet.pth"

# shadow generation model 
wget -O libcom/libcom/shadow_generation/pretrained_models/Shadow_cldm.pth "https://huggingface.co/BCMIZB/Libcom_pretrained_models/resolve/main/Shadow_cldm.pth"
wget -O libcom/libcom/shadow_generation/pretrained_models/Shadow_ppp.pth "https://huggingface.co/BCMIZB/Libcom_pretrained_models/resolve/main/Shadow_ppp.pth"


wget -O openai-clip-vit-large-patch14.zip "https://huggingface.co/BCMIZB/Libcom_pretrained_models/resolve/main/openai-clip-vit-large-patch14.zip"
unzip openai-clip-vit-large-patch14.zip 
cp -r openai-clip-vit-large-patch14 libcom/libcom/shared_pretrained_models
rm -r openai-clip-vit-large-patch14*

echo "models downloaded"