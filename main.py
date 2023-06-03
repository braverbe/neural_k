import tkinter as tk
from tkinter import filedialog

def train():
    import tensorflow as tf
    try:
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in
         tf.config.experimental.list_physical_devices("GPU")]
    except:
        pass

    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

    from mltu.tensorflow.dataProvider import DataProvider
    from mltu.tensorflow.losses import CTCloss
    from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
    from mltu.tensorflow.metrics import CWERMetric

    from mltu.preprocessors import ImageReader
    from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding
    from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate
    from mltu.annotations.images import CVImage

    from model import train_model
    from configs import ModelConfigs

    import os
    from urllib.request import urlopen
    from io import BytesIO
    from zipfile import ZipFile

    def download_and_unzip(url, extract_to="Datasets"):
        http_response = urlopen(url)
        zipfile = ZipFile(BytesIO(http_response.read()))
        zipfile.extractall(path=extract_to)

    if not os.path.exists(os.path.join("Datasets", "captcha_images_v2")):
        download_and_unzip("https://github.com/AakashKumarNain/CaptchaCracker/raw/master/captcha_images_v2.zip",
                           extract_to="Datasets")

    # Create a list of all the images and labels in the dataset
    dataset, vocab, max_len = [], set(), 0
    captcha_path = os.path.join("Datasets", "captcha_images_v2")
    for file in os.listdir(captcha_path):
        file_path = os.path.join(captcha_path, file)
        label = os.path.splitext(file)[0]  # Get the file name without the extension
        dataset.append([file_path, label])
        vocab.update(list(label))
        max_len = max(max_len, len(label))

    configs = ModelConfigs()

    # Save vocab and maximum text length to configs
    configs.vocab = "".join(vocab)
    configs.max_text_length = max_len
    configs.save()

    # Create a data provider for the dataset
    data_provider = DataProvider(
        dataset=dataset,
        skip_validation=True,
        batch_size=configs.batch_size,
        data_preprocessors=[ImageReader(CVImage)],
        transformers=[
            ImageResizer(configs.width, configs.height),
            LabelIndexer(configs.vocab),
            LabelPadding(max_word_length=configs.max_text_length, padding_value=len(configs.vocab))
        ],
    )
    # Split the dataset into training and validation sets
    train_data_provider, val_data_provider = data_provider.split(split=0.9)

    # Augment training data with random brightness, rotation and erode/dilate
    train_data_provider.augmentors = [RandomBrightness(), RandomRotate(), RandomErodeDilate()]

    # Creating TensorFlow model architecture
    model = train_model(
        input_dim=(configs.height, configs.width, 3),
        output_dim=len(configs.vocab),
    )

    # Compile the model and print summary
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=configs.learning_rate),
        loss=CTCloss(),
        metrics=[CWERMetric(padding_token=len(configs.vocab))],
        run_eagerly=False
    )
    model.summary(line_length=110)
    # Define path to save the model
    os.makedirs(configs.model_path, exist_ok=True)

    # Define callbacks
    earlystopper = EarlyStopping(monitor="val_CER", patience=50, verbose=1)
    checkpoint = ModelCheckpoint(f"{configs.model_path}/model.h5", monitor="val_CER", verbose=1, save_best_only=True,
                                 mode="min")
    trainLogger = TrainLogger(configs.model_path)
    tb_callback = TensorBoard(f"{configs.model_path}/logs", update_freq=1)
    reduceLROnPlat = ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=20, verbose=1,
                                       mode="auto")
    model2onnx = Model2onnx(f"{configs.model_path}/model.h5")

    # Train the model
    model.fit(
        train_data_provider,
        validation_data=val_data_provider,
        epochs=configs.train_epochs,
        callbacks=[earlystopper, checkpoint, trainLogger, reduceLROnPlat, tb_callback, model2onnx],
        workers=configs.train_workers
    )

    # Save training and validation datasets as csv files
    train_data_provider.to_csv(os.path.join(configs.model_path, "train.csv"))
    val_data_provider.to_csv(os.path.join(configs.model_path, "val.csv"))

def choose_image():
    file_path = filedialog.askopenfilename()
    path_entry.delete(0, tk.END)
    path_entry.insert(0, file_path)

def choose_folder():
    folder_path = filedialog.askdirectory()
    folder_entry.delete(0, tk.END)
    folder_entry.insert(0, folder_path)

def main_function():
    import pandas as pd
    df = pd.read_csv("Datasets/my_created_images/val.csv")
    # print(df)
    file_path = path_entry.get().replace('/', '\\')
    # print(file_path)
    potential_value = file_path.split('\\')[-1].split('.')[0]
    # print(potential_value)
    df.at[0, "0"] = file_path
    df.at[0, "1"] = potential_value
    # print(df)
    df.to_csv("Datasets/my_created_images/val.csv", index=False)
    # df.to_csv()

    import cv2
    import typing
    import numpy as np

    from mltu.inferenceModel import OnnxInferenceModel
    from mltu.utils.text_utils import ctc_decoder, get_cer

    class ImageToWordModel(OnnxInferenceModel):
        def __init__(self, char_list: typing.Union[str, list], *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.char_list = char_list

        def predict(self, image: np.ndarray):
            image = cv2.resize(image, self.input_shape[:2][::-1])

            image_pred = np.expand_dims(image, axis=0).astype(np.float32)

            preds = self.model.run(None, {self.input_name: image_pred})[0]

            text = ctc_decoder(preds, self.char_list)[0]

            return text

    if __name__ == "__main__":
        import pandas as pd
        from tqdm import tqdm
        from mltu.configs import BaseModelConfigs

        folder_path_education = folder_entry.get().replace('/', '\\')

        configs = BaseModelConfigs.load(f"{folder_path_education}/configs.yaml")

        model = ImageToWordModel(model_path=configs.model_path, char_list=configs.vocab)

        # df = pd.read_csv("Models/02_captcha_to_text/202305301759/val.csv").values.tolist()
        df = pd.read_csv("Datasets/my_created_images/val.csv").values.tolist()

        accum_cer = []
        for image_path, label in tqdm(df):
            image = cv2.imread(image_path)

            prediction_text = model.predict(image)

            cer = get_cer(prediction_text, label)
            print(f"Image: {image_path}, Label: {label}, Prediction: {prediction_text}, CER: {cer}")

            output_label.config(text=prediction_text)

            cv2.imshow(prediction_text, image)


            accum_cer.append(cer)

        print(f"Average CER: {np.average(accum_cer)}")


root = tk.Tk()

# Создаем кнопку "Обзор"
browse_button = tk.Button(root, text="Train neural network", command=train)
browse_button.pack()

label = tk.Label(root, text="OR \nChoose training data folder:")
label.pack()

# Создаем кнопку "Обзор"
browse_button = tk.Button(root, text="Choose", command=choose_folder)
browse_button.pack()

# Создаем текстовое поле для отображения выбранной папки
folder_entry = tk.Entry(root, width=50)
folder_entry.pack()

label_image = tk.Label(root, text="Choose image:")
label_image.pack()

browse_button = tk.Button(root, text="Browse", command=choose_image)
browse_button.pack()

path_entry = tk.Entry(root, width=50)
path_entry.pack()

find_button = tk.Button(root, text="Guess image", command=main_function)
find_button.pack()

output_label = tk.Label(root, text="")
output_label.pack()

root.mainloop()