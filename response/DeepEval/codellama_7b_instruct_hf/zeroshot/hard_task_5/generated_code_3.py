import keras
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, concatenate
from keras.applications import VGG16

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Block 1
    x = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)
    x = Conv2D(64, (1, 1), activation='relu')(x[0])
    x = Conv2D(64, (1, 1), activation='relu')(x[1])
    x = Conv2D(64, (1, 1), activation='relu')(x[2])
    x = concatenate(x, axis=1)

    # Block 2
    x = Lambda(lambda x: tf.reshape(x, (x.shape[0], x.shape[1], x.shape[2], 3)))(x)
    x = Permute((2, 3, 1))(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)

    # Branch
    branch = Lambda(lambda x: tf.split(x, 3, axis=1))(input_layer)
    branch = Conv2D(64, (1, 1), activation='relu')(branch[0])
    branch = Conv2D(64, (1, 1), activation='relu')(branch[1])
    branch = Conv2D(64, (1, 1), activation='relu')(branch[2])
    branch = concatenate(branch, axis=1)

    # Addition
    x = concatenate([x, branch])

    # Output
    output = Dense(10, activation='softmax')(x)

    # Model
    model = Model(inputs=input_layer, outputs=output)

    return model