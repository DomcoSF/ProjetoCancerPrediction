import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.optimizers import Adamax
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


loaded_model = tf.keras.models.load_model(r"C:\Users\Felipe\PycharmProjects\pythonProject1\model.Xception.h5", compile=False)
loaded_model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

image_path = r'D:\MBA\DataBases\LC25000\lung_colon_image_set\lung_image_sets\lung_aca\lungaca1002.jpeg'
image = Image.open(image_path)

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

# Get the class Name from the training generator
g_dict = train_gen.class_indices

classes = list(g_dict.keys())


# Preprocess the image
img = image.resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Make predictions
predictions = loaded_model.predict(img_array)
class_labels = classes
score = tf.nn.softmax(predictions[0])
print(f"{class_labels[tf.argmax(score)]}")

predictions