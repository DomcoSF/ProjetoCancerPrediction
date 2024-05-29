# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
import time
import shutil
import pathlib
import itertools
from PIL import Image

# import data handling tools
import cv2
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style('darkgrid')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# import Deep learning Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam, Adamax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras import regularizers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Ignore Warnings
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    #Generate Model
    '''
    # Generate data paths with labels
    data_dir = r'D:\MBA\DataBases\LC25000\lung_colon_image_set'

    # Initialize empty lists to store file paths and labels

    filepaths = []
    labels = []
    # Get a list of subdirectories
    folds = os.listdir(data_dir)

    # Iterate over each fold in the dataset
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        flist = os.listdir(foldpath)
        # Iterate over each file in the current fold
        for f in flist:
            f_path = os.path.join(foldpath, f)
            filelist = os.listdir(f_path)
            # Iterate over each file in the current fold
            for file in filelist:
                fpath = os.path.join(f_path, file)
                filepaths.append(fpath)
                # Determine the label based on the subdirectory (f)
                if f == 'colon_aca':
                    labels.append('Colon Adenocarcinoma')

                elif f == 'colon_n':
                    labels.append('Colon Benign Tissue')

                elif f == 'lung_aca':
                    labels.append('Lung Adenocarcinoma')

                elif f == 'lung_n':
                    labels.append('Lung Benign Tissue')

                elif f == 'lung_scc':
                    labels.append('Lung Squamous Cell Carcinoma')

    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepaths, name='filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis=1)

    # print(df)

    labels = df['labels']
    train_df, temp_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=labels)
    valid_df, test_df = train_test_split(temp_df, train_size=0.5, shuffle=True, random_state=123,
                                         stratify=temp_df['labels'])

    # Define image size, channels, and shape
    batch_size = 64
    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)

    # Create ImageDataGenerator  for training and testing
    tr_gen = ImageDataGenerator()
    ts_gen = ImageDataGenerator()

    # Training data generator
    train_gen = tr_gen.flow_from_dataframe(train_df, x_col='filepaths', y_col='labels',
                                           target_size=img_size, class_mode='categorical',
                                           color_mode='rgb', shuffle=True, batch_size=batch_size)

    # Validation data generator
    valid_gen = ts_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col='labels',
                                           target_size=img_size, class_mode='categorical',
                                           color_mode='rgb', shuffle=True, batch_size=batch_size)

    # Testing data generator
    test_gen = ts_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col='labels',
                                          target_size=img_size, class_mode='categorical',
                                          color_mode='rgb', shuffle=False, batch_size=batch_size)

    # Get the class Name from the training generator
    g_dict = train_gen.class_indices

    classes = list(g_dict.keys())

    images, labels = next(train_gen)

    plt.figure(figsize=(20, 20))

    for i in range(16):
        plt.subplot(4, 4, i + 1)

        image = images[i] / 255

        plt.imshow(image)

        index = np.argmax(labels[i])

        class_name = classes[index]

        plt.title(class_name, color='red', fontsize=12)

        plt.axis('off')


    #plt.show()

    import tensorflow as tf
    from tensorflow.keras.applications import Xception
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
    from tensorflow.keras.optimizers import Adamax
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.models import Model

    img_size = (224, 224)
    channels = 3
    img_shape = (img_size[0], img_size[1], channels)
    class_count = len(list(train_gen.class_indices.keys()))


    def get_callbacks(model_name):
        callbacks = []

        # Use correct syntax for ModelCheckpoint
        checkpoint = ModelCheckpoint(filepath=f'model.{model_name}.h5', verbose=1, monitor='val_accuracy', mode='max')
        callbacks.append(checkpoint)

        # Import ReduceLROnPlateau if not imported earlier
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
        callbacks.append(reduce_lr)

        # Import EarlyStopping if not imported earlier
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
        callbacks.append(early_stopping)

        return callbacks

    # Create Xception base model


    base_model = Xception(input_shape=img_shape, include_top=False, weights='imagenet')

    base_model.trainable = True

    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    y = Dense(256, activation='relu')(x)

    predictions = Dense(class_count, activation='softmax', name='final')(y)

    model_Xception = Model(inputs=base_model.input, outputs=predictions)

    model_Xception.compile(optimizer=Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model_Xception.summary())

    callbacks = get_callbacks('Xception')
    history_Xception = model_Xception.fit(train_gen, epochs=10, validation_data=valid_gen, callbacks=[callbacks])

    tr_acc = history_Xception.history['accuracy']
    tr_loss = history_Xception.history['loss']
    val_acc = history_Xception.history['val_accuracy']
    val_loss = history_Xception.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i + 1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    # Plot training history
    plt.figure(figsize=(20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label='Training loss')
    plt.plot(Epochs, val_loss, 'g', label='Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s=150, c='blue', label=loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label='Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label='Validation Accuracy')
    plt.scatter(index_acc + 1, acc_highest, s=150, c='blue', label=acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()

    ts_length = len(test_df)
    # Choose a suitable test batch size (smaller than the training batch size)
    test_batch_size = min(64, ts_length)
    test_steps = ts_length // test_batch_size

    train_score = model_Xception.evaluate(train_gen, steps=test_steps, verbose=1)
    valid_score = model_Xception.evaluate(valid_gen, steps=test_steps, verbose=1)
    test_score = model_Xception.evaluate(test_gen, steps=test_steps, verbose=1)

    print("Train Loss: ", train_score[0])
    print("Train Accuracy: ", train_score[1])
    print('-' * 20)
    print("Valid Loss: ", valid_score[0])
    print("Valid Accuracy: ", valid_score[1])
    print('-' * 20)
    print("Test Loss: ", test_score[0])
    print("Test Accuracy: ", test_score[1])

    preds = model_Xception.predict_generator(test_gen)
    y_pred_Xception = np.argmax(preds, axis=1)

    g_dict = test_gen.class_indices
    classes = list(g_dict.keys())

    # Confusion matrix
    cm = confusion_matrix(test_gen.classes, y_pred_Xception)

    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment='center', color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.show()

    print(classification_report(test_gen.classes, y_pred_Xception, target_names=classes))'''

    # Predictions
    # Load Model
    loaded_model = tf.keras.models.load_model(r"C:\Users\Felipe\PycharmProjects\pythonProject1\model.Xception.h5",
                                              compile=False)
    loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    #Load image
    image_path = r"D:\Imagens\Captura de tela 2024-02-11 205821.jpg"
    image = Image.open(image_path)

    # Preprocess the image
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Make predictions
    predictions = loaded_model.predict(img_array)

    lista = ['Colon Adenocarcinoma', 'Colon Benign Tissue', 'Lung Adenocarcinoma', 'Lung Benign Tissue',
             'Lung Squamous Cell Carcinoma']

    class_labels = lista
    score = tf.nn.softmax(predictions[0])

    print(f"{class_labels[tf.argmax(score)]}")

    np.set_printoptions(suppress=True)
    print(lista)
    print(predictions)
    # Predictions

