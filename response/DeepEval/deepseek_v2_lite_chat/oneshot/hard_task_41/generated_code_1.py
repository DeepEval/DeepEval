import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D, Dropout

def dl_model():
    # Block 1: Multiple pooling paths
    input_layer = Input(shape=(28, 28, 1))
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv2 = Conv2D(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(input_layer)
    conv3 = Conv2D(filters=32, kernel_size=(4, 4), strides=(4, 4), padding='same', activation='relu')(input_layer)
    avg_pooling = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    flatten = Flatten()(avg_pooling)
    dropout = Dropout(0.5)(flatten)
    concat = Concatenate()([dropout, conv1, conv2, conv3])

    # Block 2: Branch connections for feature extraction
    dense1 = Dense(units=128, activation='relu')(concat)
    dense2 = Dense(units=64, activation='relu')(dense1)

    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(concat)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat)
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(concat)
    branch4 = AveragePooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(concat)

    branch1 = Dense(units=64, activation='relu')(branch1)
    branch2 = Dense(units=64, activation='relu')(branch2)
    branch3 = Dense(units=64, activation='relu')(branch3)
    branch4 = Dense(units=64, activation='relu')(branch4)

    output_branch1 = Dense(units=10, activation='softmax')(branch1)
    output_branch2 = Dense(units=10, activation='softmax')(branch2)
    output_branch3 = Dense(units=10, activation='softmax')(branch3)
    output_branch4 = Dense(units=10, activation='softmax')(branch4)

    # Concatenate outputs from branches
    fused_output = Concatenate()([output_branch1, output_branch2, output_branch3, output_branch4])

    # Final classification layer
    output_layer = Dense(units=10, activation='softmax')(fused_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
model.summary()