import keras
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense, tf

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block: Split-Channel Separable Convolutions
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=2))(input_layer)
    
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])

    x = Concatenate()([branch1, branch2, branch3])

    # Second Block: Multi-Branch Feature Extraction
    branch4 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    branch5 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch4)
    branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)

    branch6 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x = Concatenate()([branch4, branch5, branch6])

    # Global Average Pooling and Classification
    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model