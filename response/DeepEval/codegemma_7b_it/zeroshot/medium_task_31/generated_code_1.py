from tensorflow.keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

def dl_model():

    # Input layer for the image
    input_image = Input(shape=(32, 32, 3))

    # Split the input image into three groups along the channel dimension
    group1 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3, name='group1'))(input_image)
    group2 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3, name='group2'))(input_image)
    group3 = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3, name='group3'))(input_image)

    # Apply convolutional kernels to each group
    conv1_1 = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', name='conv1_1')(group1)
    conv1_3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', name='conv1_3')(group2)
    conv1_5 = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', name='conv1_5')(group3)

    # Max pooling layers for each group
    max_pool1_1 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='max_pool1_1')(conv1_1)
    max_pool1_3 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='max_pool1_3')(conv1_3)
    max_pool1_5 = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', name='max_pool1_5')(conv1_5)

    # Concatenate the outputs from each group
    concat = concatenate([max_pool1_1, max_pool1_3, max_pool1_5], axis=3, name='concat')

    # Flatten the concatenated features
    flatten = Flatten(name='flatten')(concat)

    # Fully connected layers for classification
    dense1 = Dense(units=512, activation='relu', name='dense1')(flatten)
    dense2 = Dense(units=10, activation='softmax', name='dense2')(dense1)

    # Create the model
    model = Model(inputs=input_image, outputs=dense2)

    return model