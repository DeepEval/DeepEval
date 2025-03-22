import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from keras.models import Model

def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    x = Input(shape=input_shape)
    x = Lambda(lambda x: tf.split(x, 3, axis=2))(x)
    x1 = Conv2D(32, (1, 1), padding='same', activation='relu')(x[0])
    x2 = Conv2D(64, (3, 3), padding='same', activation='relu')(x[1])
    x3 = Conv2D(128, (5, 5), padding='same', activation='relu')(x[2])
    x = Concatenate()([x1, x2, x3])
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Define the second block
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x1 = Conv2D(256, (1, 1), padding='same', activation='relu')(x)
    x2 = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x3 = MaxPooling2D(pool_size=(2, 2))(x)
    x = Concatenate()([x1, x2, x3])

    # Define the output layer
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=x, outputs=x)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model