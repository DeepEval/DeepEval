import tensorflow as tf
from tensorflow import keras


def dl_model():
    # Define the input shape for the model
    input_shape = (32, 32, 3)

    # Define the first block of the model
    x = keras.Input(shape=input_shape)
    x1 = keras.layers.Lambda(lambda x: tf.split(x, 3, axis=-1))(x)
    x11 = keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x1[0])
    x12 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x1[1])
    x13 = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu')(x1[2])
    x1_concat = keras.layers.Concatenate()([x11, x12, x13])

    # Define the second block of the model
    x2 = keras.Input(shape=input_shape)
    x21 = keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x2)
    x22 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x2)
    x23 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x2)
    x24 = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu')(x2)
    x2_concat = keras.layers.Concatenate()([x21, x22, x23, x24])

    # Define the third block of the model
    x3 = keras.Input(shape=input_shape)
    x31 = keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x3)
    x32 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x3)
    x33 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x3)
    x34 = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu')(x3)
    x3_concat = keras.layers.Concatenate()([x31, x32, x33, x34])

    # Define the fourth block of the model
    x4 = keras.Input(shape=input_shape)
    x41 = keras.layers.Conv2D(filters=32, kernel_size=1, activation='relu')(x4)
    x42 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x4)
    x43 = keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu')(x4)
    x44 = keras.layers.Conv2D(filters=32, kernel_size=5, activation='relu')(x4)
    x4_concat = keras.layers.Concatenate()([x41, x42, x43, x44])

    # Define the flatten layer for the output
    flatten = keras.layers.Flatten()(x_concat)

    # Define the fully connected layer for the output
    fc = keras.layers.Dense(units=10, activation='softmax')(flatten)

    # Define the model
    model = keras.Model(inputs=[x1, x2, x3, x4], outputs=fc)

    # Compile the model with a loss function and optimizer
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Return the constructed model
    return model