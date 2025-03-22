import tensorflow as tf
from keras.layers import Input, Conv2D, Lambda, MaxPooling2D, Concatenate, GlobalAveragePooling2D, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # First Block
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(input_layer)
    
    # Channel-wise separable convolutions
    branch1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0])
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1])
    branch3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2])
    
    # Concatenate outputs from branches
    x = Concatenate()([branch1, branch2, branch3])

    # Second Block
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    
    branch2_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch2_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_1)
    branch2_3 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2_2)

    branch3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(conv1)

    # Concatenate outputs from branches
    x = Concatenate()([conv1, branch2_3, branch3])

    # Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Output Layer
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model