import keras
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Add, AveragePooling2D, Flatten, Dense
from keras import initializers

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Initial convolutional layer
    conv_init = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_uniform')(input_layer)
    conv_init = BatchNormalization()(conv_init)
    conv_init = Activation('relu')(conv_init)

    # Basic block 1
    conv_basic1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_uniform')(conv_init)
    conv_basic1 = BatchNormalization()(conv_basic1)
    conv_basic1 = Activation('relu')(conv_basic1)

    conv_branch1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_uniform')(conv_basic1)
    conv_branch1 = BatchNormalization()(conv_branch1)

    conv_basic1 = Add()([conv_basic1, conv_branch1])
    conv_basic1 = Activation('relu')(conv_basic1)

    # Basic block 2
    conv_basic2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_uniform')(conv_basic1)
    conv_basic2 = BatchNormalization()(conv_basic2)
    conv_basic2 = Activation('relu')(conv_basic2)

    conv_branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_uniform')(conv_basic2)
    conv_branch2 = BatchNormalization()(conv_branch2)

    conv_basic2 = Add()([conv_basic2, conv_branch2])
    conv_basic2 = Activation('relu')(conv_basic2)

    # Branch
    conv_branch = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='he_uniform')(conv_basic2)
    conv_branch = BatchNormalization()(conv_branch)
    conv_branch = Activation('relu')(conv_branch)

    # Feature fusion
    concat_features = Add()([conv_basic2, conv_branch])

    # Average pooling
    avg_pool = AveragePooling2D(pool_size=(2, 2), strides=2, padding='valid')(concat_features)

    # Flatten
    flatten_layer = Flatten()(avg_pool)

    # Fully connected layer
    dense1 = Dense(units=128, activation='relu')(flatten_layer)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model