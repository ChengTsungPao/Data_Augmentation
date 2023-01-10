from function import get_image_paths, get_gt_paths, get_merge_path, get_image_bbox, get_gt_param
import cv2, os

def save_crop_images_txt(save_foler, image_paths, gt_paths):
    save_path = get_merge_path([save_foler, "threshold={}".format(str(THRESHOLD))])
    number_of_image = len(image_paths)

    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths)):
        filename = image_path.split("\\")[-1].split(".jpg")[0]

        classify, (x1, y1), (x2, y2), image = get_image_bbox(image_path, gt_path)
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

        print("\r", save_foler + ": %.4f" % (((i + 1) / number_of_image) * 100.), "%", end=" ")

if __name__ == "__main__":
    THRESHOLD = 0.02
    SAVE_IMAGE_TYPE = "OriginCrop"

    dataset_folder = "dataset"
    save_foler = "augmentation"
    image_paths, gt_paths = get_image_paths(dataset_folder), get_gt_paths(dataset_folder)
    save_crop_images_txt(save_foler, image_paths, gt_paths)
