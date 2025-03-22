from keras.layers import Input, Lambda, Dense, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Add
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    input_layer = Input(shape=input_shape)
    x = Lambda(tf.split, axis=3, num_or_size_splits=3)(input_layer)
    x = [Conv2D(filters=32, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'block1_conv1_{i}')(x) for i in range(3)]
    x = [BatchNormalization()(x) for x in x]
    x = Add()(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='block1_pool')(x)

    # Define the second block
    input_layer = x
    x = [Conv2D(filters=32, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'block2_conv1_{i}')(x) for i in range(3)]
    x = [BatchNormalization()(x) for x in x]
    x = Add()(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='block2_pool')(x)

    # Define the third block
    input_layer = x
    x = [Conv2D(filters=32, kernel_size=1, strides=1, padding='same', use_bias=False, name=f'block3_conv1_{i}')(x) for i in range(3)]
    x = [BatchNormalization()(x) for x in x]
    x = Add()(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=2, strides=2, padding='same', name='block3_pool')(x)

    # Define the final layer
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.2)(x)
    x = Dense(10, activation='softmax', name='fc2')(x)

    # Create the model
    model = Model(inputs=input_layer, outputs=x)

    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model