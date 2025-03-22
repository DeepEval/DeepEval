import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Dense, Flatten, Add
from tensorflow.keras.optimizers import Adam


def dl_model():
    
    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(x_train)

    # Define the input shape
    input_shape = x_train[0].shape
    input_layer = Input(shape=input_shape)

    # Block 1: Splitting and processing
    x = Conv2D(32, (1, 1), activation='relu')(input_layer)  # 1x1 conv
    x1 = Lambda(lambda x: tf.split(x, 3, axis=-1))(x)

    # Process each group with another 1x1 conv
    x = [Conv2D(32, (1, 1), activation='relu')(group) for group in x1]
    x = Add()(x)  # Concatenate along the channel dimension

    # Block 2: Channel shuffling
    shape = x.shape[-1]
    x = Conv2D(shape, (1, 1), padding='same')(x)  # Swap third and fourth dimensions
    x = tf.transpose(x, [0, 1, 3, 2])  # Swap third and fourth dimensions
    x = Conv2D(shape, (3, 3), padding='same', data_format='channels_last')(x)  # Reshape and shuffle

    # Block 3: Depthwise separable convolution
    x = DepthwiseConv2D((3, 3), activation='relu', depth_multiplier=1)(x)  # Depthwise conv
    x = Conv2D(10, (1, 1), padding='same')(x)  # 1x1 conv

    # Combine outputs
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)  # Fully connected layer

    # Define the model
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

# Create and train the model
model = dl_model()
model.fit(datagen.flow(x_train, y_train, batch_size=64), validation_data=(x_test, y_test), epochs=10)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc) 