import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize to [0, 1]
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encode labels

# Image data augmentation
datagen = ImageDataGenerator(rotation_range=15, horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
datagen.fit(x_train)

# Define the generator
def dl_model():
    # Block 1
    input_layer = Input(shape=x_train[0].shape)
    split1 = Conv2D(64, (1, 1), activation='relu')(input_layer)
    split2 = Conv2D(64, (3, 3), activation='relu')(input_layer)
    split3 = Conv2D(64, (5, 5), activation='relu')(input_layer)
    bn1 = tf.keras.layers.BatchNormalization()(split1)
    bn2 = tf.keras.layers.BatchNormalization()(split2)
    bn3 = tf.keras.layers.BatchNormalization()(split3)
    
    concat = tf.keras.layers.concatenate([bn1, bn2, bn3])
    
    # Block 2
    conv1 = Conv2D(64, (1, 1), activation='relu')(concat)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    path2_pool = MaxPooling2D(pool_size=(3, 3))(conv1)
    path2_conv = Conv2D(64, (1, 1))(path2_pool)
    path2_split = tf.keras.layers.Concatenate()([path2_conv, path2_pool])
    path3_split1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(conv1)
    path3_conv1 = Conv2D(64, (1, 3), strides=(1, 1), padding='same')(path3_split1)
    path3_conv2 = Conv2D(64, (3, 1), strides=(1, 1), padding='same')(path3_conv1)
    path3_concat = tf.keras.layers.Concatenate()([path3_conv2, path3_conv1])
    path4_conv1 = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(conv1)
    path4_conv2 = Conv2D(64, (1, 3), strides=(1, 1), padding='same')(path4_conv1)
    path4_conv3 = Conv2D(64, (3, 1), strides=(1, 1), padding='same')(path4_conv2)
    path4_concat = tf.keras.layers.Concatenate()([path4_conv3, path4_conv2])
    
    paths = [conv1, path2_conv, path3_concat, path4_concat]
    for path in paths:
        path = Conv2D(64, (1, 1), activation='relu')(path)
        path = UpSampling2D(size=(2, 2))(path)
        path = Conv2D(64, (3, 3), activation='relu')(path)
        path = UpSampling2D(size=(2, 2))(path)
        path = Conv2D(3, (1, 1), activation='sigmoid')(path)  # Assuming binary classification
    
    model = Model(inputs=input_layer, outputs=path)
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    # Model summary
    model.summary()
    
    return model

# Build the model
model = dl_model()