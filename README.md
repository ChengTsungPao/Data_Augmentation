## Data Augmentation


### what functions ?

✅ classify far image & close image <br>
✅ image size crop to square size <br>

### How to use ?

* Run data_processing.py (First)
	* Before running .py
		* setup [THRESHOLD, OUTPUT_SIZE, SAVE_IMAGE_TYPE] = [0.02, 512, "OriginCrop"]
	* After running .py
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/close_images/OriginCrop"
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/far_images/OriginCrop" <br>

* Run data_augmentation.py (Second)
	* Before running .py
		* setup [THRESHOLD, OUTPUT_SIZE, SAVE_IMAGE_TYPE, IMAGETYPE] = [0.02, 512, "CropAug"]
        * setup [IMAGETYPE] = ["close_images" or "far_images"]
	* After running .py
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/close_images/CropAug" (IMAGETYPE = "close_images")
		* save .jpg and .txt in "./augmentation/threshold=0.02-output_size=512/far_images/CropAug" (IMAGETYPE = "far_images")
