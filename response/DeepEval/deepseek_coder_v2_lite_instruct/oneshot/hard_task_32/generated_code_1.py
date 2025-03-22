import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    def specialized_block(input_tensor):
        # Depthwise Separable Convolution
        depthwise_conv = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', depthwise_constraint=None)(input_tensor)
        depthwise_conv = Dropout(0.2)(depthwise_conv)
        
        # 1x1 Convolution
        conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(depthwise_conv)
        conv1x1 = Dropout(0.2)(conv1x1)
        
        return conv1x1

    # Three branches
    branch1 = specialized_block(input_layer)
    branch2 = specialized_block(input_layer)
    branch3 = specialized_block(input_layer)

    # Concatenate the outputs of the three branches
    concatenated = Concatenate()([branch1, branch2, branch3])

    # Batch Normalization
    batch_norm = BatchNormalization()(concatenated)

    # Flatten the result
    flatten_layer = Flatten()(batch_norm)

    # Two fully connected layers
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dropout1 = Dropout(0.5)(dense1)
    dense2 = Dense(units=64, activation='relu')(dropout1)
    dropout2 = Dropout(0.5)(dense2)
    output_layer = Dense(units=10, activation='softmax')(dropout2)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()