## Data Augmentation

### what functions ?

✅ classify far image & close image <br>
✅ image size crop to square size <br>

### How to use ?

* Step1: Prepare Dataset
    * All .jpg in "./dataset/images"
    * All .txt in "./dataset/labels" <br>

* Step2: Run data_processing.py
	* Before running .py
		* setup [THRESHOLD, OUTPUT_SIZE, SAVE_IMAGE_TYPE] = [0.02, 512, "OriginCrop"]
	* After running .py
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/close_images/OriginCrop"
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/far_images/OriginCrop" <br>

* Step3: Run data_augmentation.py
	* Before running .py
		* setup [THRESHOLD, OUTPUT_SIZE, SAVE_IMAGE_TYPE, IMAGETYPE] = [0.02, 512, "CropAug"]
        * setup [RATIO] = [0.8, 0.1, 0.1] (ratio of train、val、test)
        * setup [IMAGETYPE] = ["close_images" or "far_images"]
	* After running .py
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/close_images/CropAug/train"
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/close_images/CropAug/val"
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/close_images/CropAug/test"
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/far_images/CropAug/train"
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/far_images/CropAug/val"
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/far_images/CropAug/test"<br>

### Note
* **THRESHOLD**: use "THRESHOLD" (bbox area / image area) to classify "close_images" or "far_images"
* **OUTPUT_SIZE**: resize crop image to OUTPUT_SIZE (if OUTPUT_SIZE = -1, output origin image size)
* **SAVE_IMAGE_TYPE**: setup save folder of step2 & step3
* **IMAGETYPE**: run "IMAGETYPE" (close_images" or "far_images) data augmentation
