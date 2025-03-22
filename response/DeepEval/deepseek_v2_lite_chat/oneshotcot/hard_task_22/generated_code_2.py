import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Add

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)  # CIFAR-10 images are 3x32x32

    # Input layer
    input_layer = Input(shape=input_shape)

    # Main path
    def main_path(input_tensor):
        # Split the input into three groups along the channel axis
        split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_tensor)
        split2 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_tensor)
        split3 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_tensor)

        # Multi-scale feature extraction with separable convolutional layers
        conv1 = Conv2D(64, (1, 1), padding='same', activation='relu')(split1[0])
        conv2 = Conv2D(64, (3, 3), padding='same', activation='relu')(split1[1])
        conv3 = Conv2D(64, (5, 5), padding='same', activation='relu')(split1[2])
        conv4 = Conv2D(64, (1, 1), padding='same', activation='relu')(split2[0])
        conv5 = Conv2D(64, (3, 3), padding='same', activation='relu')(split2[1])
        conv6 = Conv2D(64, (5, 5), padding='same', activation='relu')(split2[2])
        conv7 = Conv2D(64, (1, 1), padding='same', activation='relu')(split3[0])
        conv8 = Conv2D(64, (3, 3), padding='same', activation='relu')(split3[1])
        conv9 = Conv2D(64, (5, 5), padding='same', activation='relu')(split3[2])

        # Concatenate the outputs
        concat = Concatenate()(prev_output)

        # Batch normalization and flattening
        bn = BatchNormalization()(concat)
        flatten = Flatten()(bn)

        # Fully connected layers
        dense1 = Dense(128, activation='relu')(flatten)
        dense2 = Dense(64, activation='relu')(dense1)
        output_layer = Dense(10, activation='softmax')(dense2)

        return keras.Model(inputs=input_layer, outputs=output_layer)

    # Branch path
    def branch_path(input_tensor):
        # 1x1 convolutional layer to align the number of channels
        conv = Conv2D(64, (1, 1), padding='same', activation='relu')(input_tensor)
        # Add the branch path outputs to the main path outputs
        fused_output = Add()([conv, main_path(input_tensor)])
        # Batch normalization and flattening
        bn = BatchNormalization()(fused_output)
        flatten = Flatten()(bn)
        # Fully connected layers
        dense1 = Dense(128, activation='relu')(flatten)
        dense2 = Dense(64, activation='relu')(dense1)
        output_layer = Dense(10, activation='softmax')(dense2)
        return output_layer

    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=branch_path(input_layer))

    return model

# Build and print the model
model = dl_model()
model.summary()