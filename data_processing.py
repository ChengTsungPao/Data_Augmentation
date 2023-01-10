from glob import glob
import matplotlib.pylab as plt
import numpy as np
import cv2, os

def get_image_paths(folder):
    return sorted(glob(".\\{}\\images\\*.jpg".format(folder)))

def get_gt_paths(folder):
    return sorted(glob(".\\{}\\labels\\*.txt".format(folder)))

def get_merge_path(paths):
    merge_path = paths[0]
    for path in paths[1:]:
        merge_path = os.path.join(merge_path, path)
    if not os.path.exists(merge_path):
        os.makedirs(merge_path)
    return merge_path

def get_image_bbox(image_path, gt_path):

    def getCropBbox(cx, cy, shapeX, shapeY):
        shapeX, shapeY = shapeX - 1, shapeY - 1
        LU, RD = min(cx, cy), min(shapeX - cx, shapeY - cy)
        (x1, y1), (x2, y2) = (cx - LU, cy - LU), (cx + RD, cy + RD)
        LD, RU = min(x1, shapeY - y2), min(shapeX - x2, y1)
        (x1, y2), (x2, y1) = (x1 - LD, y2 + LD), (x2 + RU, y1 - RU)
        return (x1, y1), (x2, y2)

    image = cv2.imread(image_path)
    txt_f = open(gt_path, "r")

    classify, x1_ratio, y1_ratio, w_ratio, h_ratio = np.array(txt_f.readline().split("\n")[0].split(" "), float)

    shapeX, shapeY, _ = image.shape
    cx, cy = np.array([shapeX * y1_ratio, shapeY * x1_ratio], int)
    w , h  = np.array([shapeX *  h_ratio, shapeY *  w_ratio], int)
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = cx + w // 2, cy + h // 2

    (bx1, by1), (bx2, by2) = getCropBbox(cx, cy, shapeX, shapeY)
    (x1, y1), (x2, y2) = (x1 - bx1, y1 - by1), (x2 - bx1, y2 - by1)
    shapeX, shapeY = bx2 - bx1 + 1, by2 - by1 + 1
    
    # cv2.rectangle(image, (y1, x1), (y2, x2), (0, 255, 0), 20, cv2.LINE_AA)
    # show_image(image)
    # show_image(image[x1: x2 + 1, y1: y2 + 1, :])
    return classify, (x1, y1), (x2, y2), (shapeX, shapeY), image[bx1: bx2 + 1, by1: by2 + 1]

def get_gt_param(classify, x1, y1, x2, y2, shapeX, shapeY):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1 + 1, y2 - y1 + 1
    y1_ratio, x1_ratio = cx / shapeX, cy / shapeY
    h_ratio, w_ratio = w / shapeX, h / shapeY
    return classify, x1_ratio, y1_ratio, w_ratio, h_ratio

def save_images_txt(save_foler, image_paths, gt_paths):
    save_path = get_merge_path([save_foler, "threshold={}".format(str(THRESHOLD))])
    number_of_image = len(image_paths)

    for i, (image_path, gt_path) in enumerate(zip(image_paths, gt_paths)):
        filename = image_path.split("\\")[-1].split(".jpg")[0]

        classify, (x1, y1), (x2, y2), (shapeX, shapeY), image = get_image_bbox(image_path, gt_path)
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
    
def show_image(image, scale = 8):
    shape = image.shape
    resize_shape = (shape[1] // scale, shape[0] // scale)
    resize_image = cv2.resize(image, resize_shape, interpolation=cv2.INTER_AREA)
    cv2.imshow("image", resize_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    THRESHOLD = 0.02
    SAVE_IMAGE_TYPE = "OriginCrop"

    dataset_folder = "dataset"
    save_foler = "augmentation"
    image_paths, gt_paths = get_image_paths(dataset_folder), get_gt_paths(dataset_folder)
    save_images_txt(save_foler, image_paths, gt_paths)
    