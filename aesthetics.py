import numpy as np
import os
import torch
import torch.nn as nn
from os.path import expanduser  # pylint: disable=import-outside-toplevel
from urllib.request import urlretrieve  # pylint: disable=import-outside-toplevel


def get_aesthetic_model(clip_model="vit_l_14"):
    """load the aethetic model"""
    home = expanduser("~")
    cache_folder = home + "/.cache/emb_reader"
    path_to_model = cache_folder + "/sa_0_4_"+clip_model+"_linear.pth"
    if not os.path.exists(path_to_model):
        os.makedirs(cache_folder, exist_ok=True)
        url_model = (
            "https://github.com/LAION-AI/aesthetic-predictor/blob/main/sa_0_4_"+clip_model+"_linear.pth?raw=true"
        )
        urlretrieve(url_model, path_to_model)
    if clip_model == "vit_l_14":
        m = nn.Linear(768, 1)
    elif clip_model == "vit_b_32":
        m = nn.Linear(512, 1)
    else:
        raise ValueError()
    s = torch.load(path_to_model)
    m.load_state_dict(s)
    m.eval()
    return m


device = "cuda" if torch.cuda.is_available() else "cpu"

amodel= get_aesthetic_model(clip_model="vit_l_14").to(device)




import torch
from PIL import Image
import open_clip

torch.autograd.set_detect_anomaly(True)

model, preprocess, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai', device=device)

model = model.to(device)

'''good_img = preprocess(Image.open("good4.png")).unsqueeze(0).to(device)
bad_img = preprocess(Image.open("bad4.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(good_img)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    prediction = amodel(image_features)
    print(f"Good predictions = {prediction.item()}")

with torch.no_grad():
    image_features = model.encode_image(bad_img)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    prediction = amodel(image_features)
    print(f"Bad predictions = {prediction.item()}")
'''



for param in model.parameters():
    param.requires_grad = True



def get_overlay(img_name):
    pil_image = Image.open(img_name)
    image = preprocess(pil_image).unsqueeze(0)
    np_image_resized = image.clone().detach().cpu().view(3, 224, 224).permute(1, 2, 0).numpy()

    np_image_resized = (np_image_resized - np_image_resized.min()) / (np_image_resized.max() - np_image_resized.min())

    image.requires_grad_(True)

    image = image.to(device)


    img = image.clone().detach().requires_grad_(True)

    image_features = model.encode_image(img)
    norm = image_features.norm(dim=-1, keepdim=True)

    norm_image_features = image_features / norm


    prediction = amodel(norm_image_features)
    prediction.backward()

    gradients = img.grad

    normalized_gradients = gradients / (gradients.abs().max() + 1e-8)
    normalized_gradients = normalized_gradients.abs()

    # make norm grads on log scale 
    normalized_gradients = torch.log(normalized_gradients + 1.0)

    # make them between 0 and 1
    normalized_gradients = (normalized_gradients - normalized_gradients.min()) / (normalized_gradients.max() - normalized_gradients.min())


    normalized_gradients = normalized_gradients.cpu().view(3, 224, 224).permute(1, 2, 0).numpy()

    # overlay normalized gradients on the pil_image_resized
    normalized_gradients = (normalized_gradients - normalized_gradients.min()) / (normalized_gradients.max() - normalized_gradients.min())


    overlay_image = np_image_resized * 0.8 + normalized_gradients * 0.7
    overlay_image = np.clip(overlay_image, 0, 1)

    overlay_image = Image.fromarray((overlay_image * 255).astype(np.uint8))

    return overlay_image


img_name = "good4.png"
label = img_name.split('.')[0]

overlay_image = get_overlay(img_name)
overlay_image.save(f"overlay_aes_{label}.png")

