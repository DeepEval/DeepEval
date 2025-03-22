from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, BatchNormalization, Activation, Lambda, Dense, Flatten
from tensorflow.keras.initializers import HeNormal

def dl_model():
    # Input layer
    inputs = Input(shape=(32, 32, 3))

    # First block
    x = Conv2D(filters=16, kernel_size=3, strides=1, padding='same', kernel_initializer=HeNormal())(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=HeNormal())(x)
    x = BatchNormalization()(x)
    branch_path = x

    # Main path
    main_path = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=HeNormal())(inputs)
    main_path = BatchNormalization()(main_path)
    main_path = Activation('relu')(main_path)
    main_path = Dropout(rate=0.2)(main_path)
    main_path = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', kernel_initializer=HeNormal())(main_path)
    main_path = BatchNormalization()(main_path)

    # Combine main and branch paths
    output = Add()([main_path, branch_path])

    # Second block
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=-1))(output)

    # Group 1
    x1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer=HeNormal())(x[0])
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(rate=0.2)(x1)
    x1 = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', kernel_initializer=HeNormal())(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Dropout(rate=0.2)(x1)
    x1 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer=HeNormal())(x1)
    x1 = BatchNormalization()(x1)

    # Group 2
    x2 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer=HeNormal())(x[1])
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(rate=0.2)(x2)
    x2 = Conv2D(filters=64, kernel_size=5, strides=1, padding='same', kernel_initializer=HeNormal())(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = Dropout(rate=0.2)(x2)
    x2 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer=HeNormal())(x2)
    x2 = BatchNormalization()(x2)

    # Group 3
    x3 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer=HeNormal())(x[2])
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(rate=0.2)(x3)
    x3 = Conv2D(filters=64, kernel_size=7, strides=1, padding='same', kernel_initializer=HeNormal())(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    x3 = Dropout(rate=0.2)(x3)
    x3 = Conv2D(filters=32, kernel_size=1, strides=1, padding='same', kernel_initializer=HeNormal())(x3)
    x3 = BatchNormalization()(x3)

    # Concatenate groups
    merged = Concatenate()([x1, x2, x3])

    # Output layer
    output = Flatten()(merged)
    output = Dense(units=32, kernel_initializer=HeNormal())(output)
    output = BatchNormalization()(output)
    output = Activation('relu')(output)
    output = Dropout(rate=0.2)(output)
    output = Dense(units=10, activation='softmax', kernel_initializer=HeNormal())(output)

    # Create model
    model = Model(inputs=inputs, outputs=output)

    return model