import keras
from keras.layers import Input, MaxPooling2D, Flatten, Dropout, Dense, Lambda, Concatenate, SeparableConv2D
import tensorflow as tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 RGB images

    # First block with different scales of max pooling
    pooling1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='valid')(input_layer)
    pooling2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(input_layer)
    pooling3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')(input_layer)

    # Flatten each pooling output
    flat1 = Flatten()(pooling1)
    flat2 = Flatten()(pooling2)
    flat3 = Flatten()(pooling3)

    # Apply dropout to mitigate overfitting
    drop1 = Dropout(0.5)(flat1)
    drop2 = Dropout(0.5)(flat2)
    drop3 = Dropout(0.5)(flat3)

    # Concatenate the flattened vectors
    concat_flat = Concatenate()([drop1, drop2, drop3])
    
    # Fully connected layer to transform into a four-dimensional tensor
    dense = Dense(units=512, activation='relu')(concat_flat)
    reshape_layer = Lambda(lambda x: tf.reshape(x, (-1, 8, 8, 8)))(dense)  # Reshape to (batch_size, 8, 8, 8)

    # Second block with separable convolutions
    split_tensor = Lambda(lambda x: tf.split(x, num_or_size_splits=4, axis=-1))(reshape_layer)

    # Process each group with separable convolutions
    conv1 = SeparableConv2D(filters=16, kernel_size=(1, 1), activation='relu', padding='same')(split_tensor[0])
    conv2 = SeparableConv2D(filters=16, kernel_size=(3, 3), activation='relu', padding='same')(split_tensor[1])
    conv3 = SeparableConv2D(filters=16, kernel_size=(5, 5), activation='relu', padding='same')(split_tensor[2])
    conv4 = SeparableConv2D(filters=16, kernel_size=(7, 7), activation='relu', padding='same')(split_tensor[3])

    # Concatenate outputs from the separable convolutions
    concat_conv = Concatenate()([conv1, conv2, conv3, conv4])

    # Flatten the concatenated output and pass through a fully connected layer for classification
    final_flatten = Flatten()(concat_conv)
    output_layer = Dense(units=10, activation='softmax')(final_flatten)  # CIFAR-10 has 10 classes

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model