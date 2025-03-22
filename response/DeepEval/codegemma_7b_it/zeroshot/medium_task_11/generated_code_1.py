from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def dl_model():

    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize pixel values between 0 and 1
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Define model
    input_layer = layers.Input(shape=(32, 32, 3))

    # Initial convolutional layer
    x = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')(input_layer)
    x = layers.Activation('relu')(x)

    # Channel attention path
    path_avg = layers.GlobalAveragePooling2D()(x)
    path_avg = layers.Dense(units=4, activation='relu')(path_avg)
    path_avg = layers.Dense(units=16, activation='sigmoid')(path_avg)
    path_avg = layers.Reshape((1, 1, 16))(path_avg)
    path_max = layers.GlobalMaxPooling2D()(x)
    path_max = layers.Dense(units=4, activation='relu')(path_max)
    path_max = layers.Dense(units=16, activation='sigmoid')(path_max)
    path_max = layers.Reshape((1, 1, 16))(path_max)
    attention = layers.Add()([path_avg, path_max])
    attention = layers.Activation('sigmoid')(attention)
    attention = layers.Multiply()([attention, x])

    # Spatial attention path
    path_avg = layers.AveragePooling2D()(attention)
    path_max = layers.MaxPooling2D()(attention)
    path_avg = layers.Dense(units=4, activation='relu')(path_avg)
    path_max = layers.Dense(units=4, activation='relu')(path_max)
    path_avg = layers.Dense(units=8, activation='sigmoid')(path_avg)
    path_max = layers.Dense(units=8, activation='sigmoid')(path_max)
    concat = layers.concatenate([path_avg, path_max], axis=1)
    concat_bn = layers.BatchNormalization()(concat)
    attention_spatial = layers.Activation('sigmoid')(concat_bn)
    attention_spatial = layers.Reshape((16, 1, 1))(attention_spatial)

    # Final output
    concat_attention = layers.Multiply()([attention_spatial, attention])
    flatten = layers.Flatten()(concat_attention)
    dense = layers.Dense(units=10, activation='softmax')(flatten)

    model = models.Model(inputs=input_layer, outputs=dense)

    return model