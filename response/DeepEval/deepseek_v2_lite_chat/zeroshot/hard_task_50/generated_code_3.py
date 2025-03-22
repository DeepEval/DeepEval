import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, MaxPooling2D, Flatten, Dense, concatenate, Lambda
from tensorflow.keras.optimizers import Adam

# Load and preprocess CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the input shape (32x32 images)
input_shape = (32, 32, 3)

# Generator for data augmentation
datagen = ImageDataGenerator(rotation=40, width_shift_range=0.2, height_shift_range=0.2, horizontal_flip=True)
datagen.fit(x_train)

# Functional model with two blocks
def dl_model():
    # First block
    input_layer = Input(shape=input_shape)
    x = datagen.flow(x_train, training=True).next()[0]  # Apply data augmentation
    
    # Three max pooling layers with different scales
    for i in range(3):
        x = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(x)
        if i != 2:
            x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)
        if i != 2:
            x = MaxPooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid')(x)

    x = Flatten()(x)
    x = Dropout(0.5)(x)  # Dropout before concatenation

    # Second block
    y = Input(shape=(4,))  # Four groups
    group_1, group_2, group_3, group_4 = Lambda(lambda v: tf.split(v, num_or_size_splits=4, axis=-1))(y)
    
    # Separate convolutional layers for each group
    x = group_1 * SeparableConv2D(filters=64, kernel_size=(1, 1), padding='same')(x)
    x = group_2 * SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same')(x)
    x = group_3 * SeparableConv2D(filters=64, kernel_size=(5, 5), padding='same')(x)
    x = group_4 * SeparableConv2D(filters=64, kernel_size=(7, 7), padding='same')(x)

    # Concatenate feature maps from all groups
    x = concatenate([x, x, x, x])

    # Fully connected layer and reshape
    x = Dense(512, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)  # Assuming 10 classes for CIFAR-10

    # Model
    model = Model(inputs=[input_layer, y], outputs=x)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Instantiate the model
model = dl_model()

# Print model summary
model.summary()