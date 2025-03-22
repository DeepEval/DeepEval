from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.applications import VGG16

def dl_model():
    # CIFAR-10 input shape
    input_shape = (32, 32, 3)

    # Encapsulate the tf.split operation within a Lambda layer
    split_layer = Lambda(lambda x: tf.split(x, 3, axis=-1))(Input(shape=input_shape))

    # Depthwise separable convolutional layers with different kernel sizes
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_layer[0])
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_layer[1])
    conv3 = Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_layer[2])

    # Batch normalization layer
    bn_layer = BatchNormalization()(Concatenate()([conv1, conv2, conv3]))

    # Max pooling layer
    pool_layer = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(bn_layer)

    # First block
    first_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(pool_layer)
    first_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block)
    first_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(first_block)

    # Second block
    second_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(first_block)
    second_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_block)
    second_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(second_block)

    # Third block
    third_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(second_block)
    third_block = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(third_block)
    third_block = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(third_block)

    # Flatten the output and add a dense layer
    flatten_layer = Flatten()(third_block)
    dense_layer = Dense(units=128, activation='relu')(flatten_layer)
    dense_layer = Dense(units=64, activation='relu')(dense_layer)

    # Add the final classification layer
    output_layer = Dense(units=10, activation='softmax')(dense_layer)

    # Create the model
    model = Model(inputs=Input(shape=input_shape), outputs=output_layer)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model