import os, json, time
import numpy as np
import torch
from torchvision import models, transforms
from torchvision.models.resnet import resnet50, ResNet50_Weights
from PIL import Image

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = torch.nn.Sequential(*(list(model.children())[:-1]))

# Load the pre-trained model
def load(img):
    img = img.resize((224, 224))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)

    if img.shape[-3] == 1:
        img = img.repeat(1, 3, 1, 1)
    if img.shape[-3] > 3:
        img = img[:, :3, :, :]

    return model(img)

def get_embedding_shape_from_json(file: str) -> tuple:
    with open(file, 'r') as f:
        image_id_maps = json.load(f)
    shape = (max(image_id_maps.values()) + 1, 2048)
    return shape

def process_images(base_img_dir: str, json_file : str, save_file_path: str = 'data/embeddings/images/test_image_embeddings.npy') -> int:
    # Dictionary to hold image ID and its corresponding embedding
    start_time = time.time()
    with open(json_file, 'r') as f:
        image_id_maps = json.load(f)

    embedding_shape = get_embedding_shape_from_json(json_file)
    embeddings = np.zeros(embedding_shape, dtype=np.float16)

    cnt = 0
    # print(image_id_maps.items())
    for img_path, img_id in image_id_maps.items():
        # print(img_path, img_id)
        # break
        img = Image.open(base_img_dir + img_path)
        embedding = load(img)
        embedding = embedding.squeeze().detach().numpy()
        # print(embedding.shape)
        embeddings[img_id] = embedding

        if cnt % 500 == 0:
            print(f'Processed {cnt} images; Saving embeddings...')
            np.save(save_file_path, embeddings)
        # if cnt == 10:
        #     break
        cnt += 1
    # print(embeddings.keys())
    np.save(save_file_path, embeddings)
    print("Successfully saved all the embeddings to ", save_file_path)
    end_time = time.time()
    print(f'Processing time: {end_time- start_time}')
    # ~ 1094.6 s for 10000 images
    return end_time - start_time


train_images_dir = 'data/images/train/images-qa/'
train_image_ids_map = 'data/train_image_id_mapping.json'
train_embeddings_path = 'data/embeddings/images/train_image_embeddings.npy'

test_images_dir = 'data/images/test/images-qa/'
test_image_ids_map = 'data/test_image_id_mapping.json'
test_embeddings_path = 'data/embeddings/images/test_image_embeddings.npy'

# embeddings = process_images(base_img_dir='data/images/test/images-qa/', json_file='data/test_image_id_mapping.json', save_file_path='data/embeddings/images/test_image_embeddings.npy')
embeddings = process_images(base_img_dir=train_images_dir, json_file=train_image_ids_map, save_file_path=train_embeddings_path)

# print(get_embedding_shape_from_json('data/train_image_id_mapping.json'))





