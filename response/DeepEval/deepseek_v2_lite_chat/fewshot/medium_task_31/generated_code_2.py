import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Lambda, Flatten, Concatenate, Dense, Reshape
from keras.models import Model

def dl_model():
    # Input shape
    input_shape = (32, 32, 3)  # Assuming input images are 32x32 and have 3 color channels

    # First path for multi-scale feature extraction
    def block_1(input_tensor):
        split1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split1[0])
        conv1_2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[1])
        conv1_3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split1[2])
        pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(Concatenate()([conv1_1, conv1_2, conv1_3]))
        return pool1

    # Second path for multi-scale feature extraction
    def block_2(input_tensor):
        split2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv2_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split2[0])
        conv2_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[1])
        conv2_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split2[2])
        pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(Concatenate()([conv2_1, conv2_2, conv2_3]))
        return pool2

    # Third path for multi-scale feature extraction
    def block_3(input_tensor):
        split3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(input_tensor)
        conv3_1 = Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split3[0])
        conv3_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split3[1])
        conv3_3 = Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split3[2])
        pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(Concatenate()([conv3_1, conv3_2, conv3_3]))
        return pool3

    # Concatenate outputs from all paths
    concat_layer = Concatenate()([block_1(input_tensor), block_2(input_tensor), block_3(input_tensor)])

    # Flatten and pass through fully connected layers
    flatten = Flatten()(concat_layer)
    dense1 = Dense(units=512, activation='relu')(flatten)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    # Construct the model
    model = Model(inputs=input_tensor, outputs=output_layer)

    return model

# Instantiate and return the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()