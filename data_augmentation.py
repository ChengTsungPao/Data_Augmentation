from function import get_image_paths, get_gt_paths, get_merge_path, get_image_bbox, get_gt_param
import numpy as np
import cv2, os

def flip_or_rotated(x1, y1, x2, y2, image, augType = -1):
    shapeX, shapeY, _ = image.shape
    random_kind = augType if augType >= 0 else np.random.randint(0, 5 + 1)

    if random_kind == 1:
        image = cv2.flip(image, 0)

        x1, x2 = shapeX - x1, shapeX- x2

    elif random_kind == 2:
        image = cv2.flip(image, 1)

        y1, y2 = shapeY - y1, shapeY- y2

    elif random_kind == 3:
        image = cv2.rotate(image, cv2.ROTATE_180)

        x1, x2 = shapeX - x1, shapeX- x2
        y1, y2 = shapeY - y1, shapeY- y2

    elif random_kind == 4:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        x1, x2 = shapeX - x1, shapeX- x2
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    elif random_kind == 5:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        y1, y2 = shapeY - y1, shapeY- y2
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    return x1, y1, x2, y2, image

def get_aug_image_bbox(image_path, gt_path, augType = -1):
    classify, (x1, y1), (x2, y2), image = get_image_bbox(image_path, gt_path)
    x1, y1, x2, y2, image = flip_or_rotated(x1, y1, x2, y2, image, augType)
    (x1, y1), (x2, y2) = (min(x1, x2), min(y1, y2)), (max(x1, x2), max(y1, y2))
    return classify, (x1, y1), (x2, y2), image

def save_aug_images_and_txt(save_path, image_paths, gt_paths):
    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths)):
        filename = image_path.split("\\")[-1].split(".jpg")[0]

        for augType in range(3 + 1):

            classify, (x1, y1), (x2, y2), image = get_aug_image_bbox(image_path, gt_path, augType)
            shapeX, shapeY, _ = image.shape

            save_images_path = get_merge_path([save_path, "images"])
            save_labels_path = get_merge_path([save_path, "labels"])
            save_bboxImages_path = get_merge_path([save_path, "bbox"])

            cv2.imwrite(os.path.join(save_images_path, "{}_{}.jpg".format(filename, augType)), image)

            classify, x1_ratio, y1_ratio, w_ratio, h_ratio = get_gt_param(classify, x1, y1, x2, y2, shapeX, shapeY)
            open(os.path.join(save_labels_path, "{}_{}.txt".format(filename, augType)), "w").write("{} {} {} {} {}".format(classify, x1_ratio, y1_ratio, w_ratio, h_ratio))

            cv2.rectangle(image, (y1, x1), (y2, x2), (0, 255, 0), 5, cv2.LINE_AA)
            cv2.imwrite(os.path.join(save_bboxImages_path, "{}_{}.jpg".format(filename, augType)), image)

            print("\r", save_path + ": %.4f" % (((i + 1) / len(image_paths)) * 100.), "%", end=" ")

if __name__ == "__main__":
    THRESHOLD = 0.02
    OUTPUT_SIZE = 512
    SAVE_IMAGE_TYPE = "CropAug"
    IMAGETYPE = "close_images" # "far_images"
    RATIO = [0.8, 0.1, 0.1]

    crop_path = get_merge_path(["augmentation", f"threshold={THRESHOLD}-output_size={OUTPUT_SIZE}", IMAGETYPE, "OriginCrop"])
    image_paths, gt_paths = get_image_paths(crop_path), get_gt_paths(crop_path)
    number_of_data = len(image_paths)

    # train dataset
    i, j = np.array([number_of_data * sum(RATIO[:0]), number_of_data * sum(RATIO[:1])], int)
    save_path = get_merge_path(["augmentation", f"threshold={THRESHOLD}-output_size={OUTPUT_SIZE}", IMAGETYPE, "CropAug", "train"])
    save_aug_images_and_txt(save_path, image_paths[i : j], gt_paths[i : j])
    print("\n===== train dataset augmentation finish =====")

    # val dataset
    i, j = np.array([number_of_data * sum(RATIO[:1]), number_of_data * sum(RATIO[:2])], int)
    save_path = get_merge_path(["augmentation", f"threshold={THRESHOLD}-output_size={OUTPUT_SIZE}", IMAGETYPE, "CropAug", "val"])
    save_aug_images_and_txt(save_path, image_paths[i : j], gt_paths[i : j])
    print("\n===== val dataset augmentation finish =====")

    # test dataset
    i, j = np.array([number_of_data * sum(RATIO[:2]), number_of_data * sum(RATIO[:3])], int)
    save_path = get_merge_path(["augmentation", f"threshold={THRESHOLD}-output_size={OUTPUT_SIZE}", IMAGETYPE, "CropAug", "test"])
    save_aug_images_and_txt(save_path, image_paths[i : j], gt_paths[i : j])
    print("\n===== test dataset augmentation finish =====")
