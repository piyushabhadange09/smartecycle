from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import os

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

src = "ml/data"  # each subfolder
for cls in os.listdir(src):
    folder = os.path.join(src, cls)
    if not os.path.isdir(folder): continue
    images = os.listdir(folder)
    for fn in images:
        path = os.path.join(folder, fn)
        img = load_img(path)  # PIL
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        i = 0
        for batch in datagen.flow(
            x,
            batch_size=1,
            save_to_dir=folder,
            save_prefix='aug',
            save_format='jpg'
        ):
            i += 1
            if i >= 2:  # create 2 augmentations per image
                break
