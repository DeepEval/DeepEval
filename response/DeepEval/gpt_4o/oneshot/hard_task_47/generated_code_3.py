import keras
from keras.layers import Input, Lambda, SeparableConv2D, BatchNormalization, Concatenate, Dense, AveragePooling2D, Flatten
from keras import backend as K

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First block: using Lambda to split channels
    def split_channels(x):
        return K.tf.split(x, num_or_size_splits=3, axis=-1)

    # Apply the channel split
    split_layer = Lambda(split_channels)(input_layer)

    # Feature extraction using depthwise separable conv layers
    path1 = SeparableConv2D(filters=32, kernel_size=(1, 1), activation='relu')(split_layer[0])
    path2 = SeparableConv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(split_layer[1])
    path3 = SeparableConv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(split_layer[2])

    # Concatenate outputs and apply batch normalization
    concat1 = Concatenate()([path1, path2, path3])
    batch_norm1 = BatchNormalization()(concat1)

    # Second block: multiple branches
    # Branch 1
    branch1 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(batch_norm1)
    branch1 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch1)

    # Branch 2
    branch2 = SeparableConv2D(filters=64, kernel_size=(1, 1), activation='relu')(batch_norm1)
    branch2 = SeparableConv2D(filters=64, kernel_size=(1, 7), activation='relu', padding='same')(branch2)
    branch2 = SeparableConv2D(filters=64, kernel_size=(7, 1), activation='relu', padding='same')(branch2)
    branch2 = SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same')(branch2)

    # Branch 3
    branch3 = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(batch_norm1)

    # Concatenate all branches
    concat2 = Concatenate()([branch1, branch2, branch3])

    # Fully connected layers for classification
    flatten_layer = Flatten()(concat2)
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    output_layer = Dense(units=10, activation='softmax')(dense1)

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model