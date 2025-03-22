import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Reshape, Permute
from keras.utils import to_categorical

def dl_model():  

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)  

    # Block 1
    x = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(x)
    x = [Conv2D(int(x.shape[3]/3), (1, 1), activation='relu')(branch) for branch in x]
    x = Concatenate(axis=3)(x)

    # Block 2
    x = Lambda(lambda x: tf.shape(x[0]))(x)
    x = Reshape((x[0][0], x[0][1], 3, int(x[0][3]/3)))(x)
    x = Permute((2, 3, 1, 0))(x)
    x = Reshape((x.shape[2], x.shape[3], x.shape[1], x.shape[0]))(x)

    # Block 3
    x = DepthwiseSeparableConv2D(128, (3, 3), activation='relu', padding='same')(x)

    # Branch Path
    branch = MaxPooling2D((2, 2), strides=(2, 2))(input_layer)
    branch = Flatten()(branch)
    branch = Dense(128, activation='relu')(branch)

    # Concatenate
    x = Concatenate()([x, branch])

    # Final Classification
    x = Flatten()(x)
    x = Dense(10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model