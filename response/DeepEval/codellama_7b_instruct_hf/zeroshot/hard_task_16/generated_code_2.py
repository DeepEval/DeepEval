import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, Split, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)
    x1 = Conv2D(64, (1, 1), activation='relu')(x[0])
    x2 = Conv2D(64, (3, 3), activation='relu')(x[1])
    x3 = Conv2D(64, (1, 1), activation='relu')(x[2])
    x = Concatenate()([x1, x2, x3])
    x = MaxPooling2D((2, 2))(x)

    # Transition convolution
    x = Conv2D(32, (1, 1), activation='relu')(x)
    x = Flatten()(x)

    # Block 2
    x = Dense(128, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10)(x)

    # Branch
    branch = Conv2D(10, (1, 1), activation='relu')(input_layer)
    branch = Flatten()(branch)

    # Main path
    main_path = x

    # Addition
    output = Add()([main_path, branch])

    # Output
    output = Dense(10)(output)

    # Model
    model = Model(inputs=input_layer, outputs=output)

    return model