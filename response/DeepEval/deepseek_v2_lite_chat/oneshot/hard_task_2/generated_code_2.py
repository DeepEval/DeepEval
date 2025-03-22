import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    def conv_block(x, filters, kernel_size):
        # 1x1 convolution
        conv1 = Conv2D(filters, kernel_size=(1, 1), activation='relu')(x)
        # 3x3 convolution
        conv2 = Conv2D(filters, kernel_size=(3, 3), activation='relu')(conv1)
        # 1x1 convolution
        conv3 = Conv2D(filters, kernel_size=(1, 1), activation='relu')(conv2)
        return conv3

    def main_path(x):
        # Split input into three groups along the channel dimension
        split1, split2, split3 = tf.split(x, 3, axis=-1)
        # Sequential convolutions
        conv1 = conv_block(split1, 64, (1, 1))
        conv2 = conv_block(split2, 64, (3, 3))
        conv3 = conv_block(split3, 64, (1, 1))
        # Add outputs from the three groups
        add = Concatenate()([conv1, conv2, conv3])
        return add

    def classification_head(x):
        # Batch normalization and flattening
        bn = BatchNormalization()(x)
        flat = Flatten()(bn)
        # Fully connected layers for classification
        dense1 = Dense(128, activation='relu')(flat)
        dense2 = Dense(64, activation='relu')(dense1)
        output = Dense(10, activation='softmax')(dense2)
        return output

    input_layer = Input(shape=(32, 32, 3))
    # Main path using the extended conv_block
    main_output = main_path(input_layer)
    # Classification head
    classification_output = classification_head(main_output)
    # Model construction
    model = keras.Model(inputs=input_layer, outputs=classification_output)
    return model

# Create the model
model = dl_model()
model.summary()