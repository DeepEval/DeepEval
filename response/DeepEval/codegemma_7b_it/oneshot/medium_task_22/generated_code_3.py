import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer for CIFAR-10 images
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 3x3 convolutions
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_1)
    branch_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_1)

    # Branch 2: 1x1 convolutions followed by 3x3 convolutions
    branch_2 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch_2)
    branch_2 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(branch_2)

    # Branch 3: Max pooling
    branch_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(input_layer)

    # Multi-scale feature fusion block
    multi_scale_block = Concatenate()([branch_1, branch_2, branch_3])

    # Batch normalization and flattening
    bath_norm = BatchNormalization()(multi_scale_block)
    flatten_layer = Flatten()(bath_norm)

    # Fully connected layers for classification
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Create the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()

# Print the model summary
model.summary()