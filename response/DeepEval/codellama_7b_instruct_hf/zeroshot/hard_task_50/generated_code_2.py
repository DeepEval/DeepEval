import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout, Flatten, Dense, Lambda, SeparableConv2D, MaxPooling2D
from tensorflow.keras.models import Model

def dl_model():
    # Define input shape
    input_shape = (32, 32, 3)

    # Define first block
    x = Input(shape=input_shape)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)

    # Define second block
    x = Lambda(lambda x: tf.split(x, 4, axis=-1))(x)
    x = SeparableConv2D(16, (1, 1), activation='relu')(x[0])
    x = SeparableConv2D(16, (3, 3), activation='relu')(x[1])
    x = SeparableConv2D(16, (5, 5), activation='relu')(x[2])
    x = SeparableConv2D(16, (7, 7), activation='relu')(x[3])
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Define model
    model = Model(inputs=x, outputs=x)

    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model