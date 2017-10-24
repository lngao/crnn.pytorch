import os
import lmdb # install lmdb by "pip install lmdb"
import cv2
import numpy as np
import argparse
import glob

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.iteritems():
            txn.put(k, v)

def createDataset(output_dataset_path, image_path_list, label_list, lexicon_list=None, check_valid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        output_dataset_path    : LMDB output path
        image_path_list : list of image path
        label_list     : list of corresponding groundtruth texts
        lexicon_list   : (optional) list of lexicon lists
        check_valid    : if true, check the validity of every image
    """
    assert(len(image_path_list) == len(label_list))
    nSamples = len(image_path_list)
    env = lmdb.open(output_dataset_path, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in xrange(nSamples):
        imagePath = image_path_list[i]
        label = label_list[i]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'r') as f:
            imageBin = f.read()
        if check_valid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue

        imageKey = 'image-%09d' % cnt
        labelKey = 'label-%09d' % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label
        if lexicon_list:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexicon_list[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'] = str(nSamples)
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

def get_files_from_dir(file_dir, ext_list = ['png','jpg']):
    """
    :param files_dir:
    :param ext_list:
    :return:
    """
    if not os.path.exists(file_dir):
        print file_dir + "is none"
        return None

    file_list = []
    for ext in ext_list:
        file_list += glob.glob(file_dir + '*.' + ext)
    return file_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dataset_path",
        default= "/home/extend/code/crnn_pytrch/crnn.pytorch/dataset/data.lmdb",
        help= "LMDB output path"
    )

    parser.add_argument(
        "--images_path",
        default= "/home/gaolining/extend/data/crnn_image_data/",
        help= "images path"
    )

    parser.add_argument(
        "--labels_path",
        default= "/home/gaolining/extend/data/crnn_label_data/",
        help="labels path"
    )
    args = parser.parse_args()

    image_path_list = get_files_from_dir(args.images_path)
    label_path_list = get_files_from_dir(args.labels_path, ["txt"])

    image_path_list.sort()
    label_path_list.sort()
    label_list = []
    for label_path in label_path_list:
        with open(label_path) as f:
            label_list.append(f.readline())

    createDataset(args.output_dataset_path, image_path_list, label_list)

