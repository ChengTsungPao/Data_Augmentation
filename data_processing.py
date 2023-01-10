from function import get_image_paths, get_gt_paths, get_merge_path, get_image_bbox, get_gt_param
import cv2, os

def get_resize_image_bbox(classify, x1, y1, x2, y2, image):
    if OUTPUT_SIZE <= 0:
        return classify, (x1, y1), (x2, y2), image
    scale = OUTPUT_SIZE / image.shape[0]
    image = cv2.resize(image, (OUTPUT_SIZE, OUTPUT_SIZE))
    x1, y1, x2, y2 = int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)
    return classify, (x1, y1), (x2, y2), image

def save_crop_images_txt(save_path, image_paths, gt_paths):
    number_of_image = len(image_paths)

    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths)):
        filename = image_path.split("\\")[-1].split(".jpg")[0]

        classify, (x1, y1), (x2, y2), image = get_image_bbox(image_path, gt_path)
        classify, (x1, y1), (x2, y2), image = get_resize_image_bbox(classify, x1, y1, x2, y2, image)
        shapeX, shapeY, _ = image.shape

        ratio = (x2 - x1) * (y2 - y1) / (shapeX * shapeY)
        imageType = "close_images" if ratio >= THRESHOLD else "far_images"

        save_images_path = get_merge_path([save_path, imageType, SAVE_IMAGE_TYPE, "images"])
        save_labels_path = get_merge_path([save_path, imageType, SAVE_IMAGE_TYPE, "labels"])
        save_bboxImages_path = get_merge_path([save_path, imageType, SAVE_IMAGE_TYPE, "bbox"])

        cv2.imwrite(os.path.join(save_images_path, "{}.jpg".format(filename)), image)

        classify, x1_ratio, y1_ratio, w_ratio, h_ratio = get_gt_param(classify, x1, y1, x2, y2, shapeX, shapeY)
        open(os.path.join(save_labels_path, "{}.txt".format(filename)), "w").write("{} {} {} {} {}".format(classify, x1_ratio, y1_ratio, w_ratio, h_ratio))

        cv2.rectangle(image, (y1, x1), (y2, x2), (0, 255, 0), 5, cv2.LINE_AA)
        cv2.imwrite(os.path.join(save_bboxImages_path, "{}.jpg".format(filename)), image)

        print("\r", save_path + ": %.4f" % (((i + 1) / number_of_image) * 100.), "%", end=" ")

if __name__ == "__main__":
    THRESHOLD = 0.02
    OUTPUT_SIZE = -1
    SAVE_IMAGE_TYPE = "OriginCrop"

    dataset_path = "dataset"
    save_path = get_merge_path(["augmentation", f"threshold={THRESHOLD}-output_size={OUTPUT_SIZE}"])
    image_paths, gt_paths = get_image_paths(dataset_path), get_gt_paths(dataset_path)
    save_crop_images_txt(save_path, image_paths, gt_paths)
