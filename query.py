import os
import time
import shutil
import numpy as np
import pandas as pd
from img_loader import img_loader
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader


# calculate hash for query items
# return a list of tuple of filename and hash value
def calculate_query_hash(query_dir):
    # remove last layer of alexnet
    model = models.alexnet(pretrained=True).eval()
    model.classifier = nn.Sequential(*[model.classifier[i] for i in range(6)])

    # create image loader
    dataset_reader = img_loader(query_dir)
    dataloader = DataLoader(dataset_reader, batch_size=512, shuffle=False, pin_memory=True, num_workers=12)

    detections_batch = []
    for _, (batch) in enumerate(dataloader):
        # batch[0], image content
        # batch[1], label
        result = model(batch[0]).detach().cpu().numpy()
        detections_batch.append([result, batch[1]])

    query_hashes = {}
    # flatten batches
    for res_batch in detections_batch:
        for i in range(res_batch[0].shape[0]):
            filename = res_batch[1][i]
            hash_val = np.array([j for j in res_batch[0][i]], dtype='float32')
            query_hashes[filename] = hash_val

    return query_hashes


# find knn of 'query_hashes' in 'hash_csv'
def find_knn(query_hashes, hash_csv, k):
    img_hashes = pd.read_csv(hash_csv, sep=' ')
    filenames = img_hashes[img_hashes.columns[0]].values
    hash_vals = np.array(img_hashes[img_hashes.columns[1:]].values, dtype='float32')

    knn_result = {}
    for filename, hash_val in query_hashes.items():
        dist = np.linalg.norm(hash_val - hash_vals, axis=1)
        dist = list(zip(filenames, dist))
        knn_result[filename] = sorted(dist, key=lambda tup: tup[1])[:k]

    return knn_result


# copy the knn result to 'res_dir'
def copy_result(query_dir, img_dir, res_dir, knn_result: dict):
    # check existence of 'res_dir'
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    for query_filename, top_k in knn_result.items():
        # make dir for each query
        sub_path = os.path.join(res_dir, query_filename.split('.')[0])
        if os.path.isdir(sub_path):
            shutil.rmtree(sub_path)
        os.makedirs(sub_path)

        # copy the target image to result directory
        shutil.copy(os.path.join(query_dir, query_filename), os.path.join(sub_path, f'Q_{query_filename}'))

        # copy the knn result image to result directory
        i = 0
        for res_filename, _ in top_k:
            i += 1
            shutil.copy(os.path.join(img_dir, res_filename), os.path.join(sub_path, f'{i}_{res_filename}'))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    query_dir = r'C:\Users\fuchi\Desktop\query'
    origin = r'C:\Users\fuchi\Desktop\Buffer'
    result_dir = r'C:\Users\fuchi\Desktop\LSH_knn_result'
    hash_csv = r'C:\Users\fuchi\Desktop\image_vec_n_29781_d_4096.csv'

    start = time.time()

    query_hashes = calculate_query_hash(query_dir)

    hash_cal_end = time.time()
    print("Hash calculation time: ", hash_cal_end - start)

    k = 20
    knn_result = find_knn(query_hashes, hash_csv, k)

    knn_cal_end = time.time()
    print(f'KNN calculation time: {knn_cal_end - hash_cal_end}')

    copy_result(query_dir, origin, result_dir, knn_result)

    end = time.time()
    print(f'Copy time: {end - knn_cal_end}')
    print(f'Total time: {end - start}')
