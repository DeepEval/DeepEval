import keras
from keras.layers import Input, Lambda, Conv2D, BatchNormalization, Concatenate, AveragePooling2D, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Depthwise Separable Convolutions
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    
    # Apply depthwise separable convolutions with different kernel sizes to each split
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    # Apply Batch Normalization
    branch1 = BatchNormalization()(branch1)
    branch2 = BatchNormalization()(branch2)
    branch3 = BatchNormalization()(branch3)

    # Concatenate the outputs of the three branches
    x = Concatenate()([branch1, branch2, branch3])

    # Second Block: Feature Extraction Branches
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x)
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch4)
    branch5 = Conv2D(filters=64, kernel_size=(1, 7), strides=(1, 1), padding='same', activation='relu')(x)
    branch5 = Conv2D(filters=64, kernel_size=(7, 1), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch6 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    # Concatenate all branch outputs
    x = Concatenate()([branch4, branch5, branch6])

    # Flatten and dense layers for classification
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model