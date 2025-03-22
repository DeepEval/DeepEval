import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from tensorflow.keras import backend as K

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # First Block
    x = tf.split(input_layer, num_or_size_splits=3, axis=3)
    conv1 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(x[0]))
    conv2 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x[1]))
    conv3 = Lambda(lambda x: Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(x[2]))
    concat1 = Concatenate()([conv1, conv2, conv3])

    # Second Block
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(concat1)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(x)
    max_pooling = MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid')(x)

    # Additional Branches
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(max_pooling)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch3 = MaxPooling2D(pool_size=(3, 3), strides=1, padding='same')(max_pooling)
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(branch3)

    # Concatenate Branches
    concat2 = Concatenate()([branch1, branch2, branch3])

    # Global Average Pooling and Fully Connected Layer
    gap = GlobalAveragePooling2D()(concat2)
    output_layer = Dense(units=10, activation='softmax')(gap)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model