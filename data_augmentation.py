from data_processing import get_image_paths, get_gt_paths, get_merge_path, get_image_bbox, get_gt_param
import numpy as np
import cv2, os

def flip_or_rotated(x1, y1, x2, y2, image):
    shapeX, shapeY, _ = image.shape
    random_kind = 1#np.random.randint(0, 5 + 1)

    if random_kind == 1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        x1, x2 = shapeX - x1, shapeX- x2
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    elif random_kind == 2:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        y1, y2 = shapeY - y1, shapeY- y2
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    elif random_kind == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)

        x1, x2 = shapeX - x1, shapeX- x2
        y1, y2 = shapeY - y1, shapeY- y2

    elif random_kind == 4:
        image = cv2.flip(image, 0)

        x1, x2 = shapeX - x1, shapeX- x2

    elif random_kind == 5:
        image = cv2.flip(image, 1)

        y1, y2 = shapeY - y1, shapeY- y2

    return x1, y1, x2, y2, image

def get_aug_images_bboxes(save_path, image_paths, gt_paths):
    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths)):
        filename = image_path.split("\\")[-1].split(".jpg")[0]

        classify, (x1, y1), (x2, y2), (shapeX, shapeY), image = get_image_bbox(image_path, gt_path)
        x1, y1, x2, y2, image = flip_or_rotated(x1, y1, x2, y2, image)
        save_param = [classify, (x1, y1), (x2, y2), image]

        save_images_and_txt(save_path, filename, save_param)
        print("\r", save_path + ": %.4f" % (((i + 1) / len(image_paths)) * 100.), "%", end=" ")

def save_images_and_txt(save_path, filename, save_param):
    classify, (x1, y1), (x2, y2), image = save_param
    shapeX, shapeY, _ = image.shape

    save_images_path = get_merge_path([save_path, "images"])
    save_labels_path = get_merge_path([save_path, "labels"])
    save_bboxImages_path = get_merge_path([save_path, "bbox"])

    cv2.imwrite(os.path.join(save_images_path, "{}.jpg".format(filename)), image)

    classify, x1_ratio, y1_ratio, w_ratio, h_ratio = get_gt_param(classify, x1, y1, x2, y2, shapeX, shapeY)
    open(os.path.join(save_labels_path, "{}.txt".format(filename)), "w").write("{} {} {} {} {}".format(classify, x1_ratio, y1_ratio, w_ratio, h_ratio))

    cv2.rectangle(image, (y1, x1), (y2, x2), (0, 255, 0), 5, cv2.LINE_AA)
    cv2.imwrite(os.path.join(save_bboxImages_path, "{}.jpg".format(filename)), image)

if __name__ == "__main__":
    THRESHOLD = 0.02
    SAVE_IMAGE_TYPE = "CropAug"
    IMAGETYPE = "close_images" # "far_images"

    crop_path = get_merge_path(["augmentation", "threshold={}".format(str(THRESHOLD)), IMAGETYPE, "OriginCrop"])
    save_path = get_merge_path(["augmentation", "threshold={}".format(str(THRESHOLD)), IMAGETYPE, "CropAug"])
    image_paths, gt_paths = get_image_paths(crop_path), get_gt_paths(crop_path)
    get_aug_images_bboxes(save_path, image_paths, gt_paths)