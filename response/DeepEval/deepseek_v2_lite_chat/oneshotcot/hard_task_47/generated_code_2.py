import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Activation
from keras.models import Model
from keras.layers import AveragePooling2D, ZeroPadding2D

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # First block for feature extraction
    split1 = Lambda(lambda x: keras.backend.split(x, 3, axis=-1))(input_layer)
    split1[0] = Conv2D(64, (1, 1), padding='same')(split1[0])
    split1[1] = Conv2D(64, (3, 3), padding='same')(split1[1])
    split1[2] = Conv2D(64, (5, 5), padding='same')(split1[2])
    split1 = BatchNormalization()(split1)
    pool1 = MaxPooling2D(pool_size=(1, 1))(split1[0])
    pool2 = MaxPooling2D(pool_size=(3, 3))(split1[1])
    pool3 = MaxPooling2D(pool_size=(5, 5))(split1[2])
    pool1 = Conv2D(64, (1, 1), padding='same')(pool1)
    pool2 = Conv2D(64, (1, 1), padding='same')(pool2)
    pool3 = Conv2D(64, (1, 1), padding='same')(pool3)
    pool1 = BatchNormalization()(pool1)
    pool2 = BatchNormalization()(pool2)
    pool3 = BatchNormalization()(pool3)
    concat = Concatenate(axis=-1)([pool1, pool2, pool3])

    # Second block for feature extraction
    branch1 = Conv2D(64, (1, 1), padding='same', activation='relu')(input_layer)
    branch2 = Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)
    branch3 = AveragePooling2D((1, 7), strides=(1, 1), padding='valid')(input_layer)
    branch3 = AveragePooling2D((7, 1), strides=(1, 1), padding='valid')(branch3)
    branch3 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch3)
    branch4 = MaxPooling2D((5, 1), strides=(1, 1), padding='valid')(input_layer)
    branch4 = MaxPooling2D((1, 7), strides=(1, 1), padding='valid')(branch4)
    branch4 = MaxPooling2D((7, 1), strides=(1, 1), padding='valid')(branch4)
    branch4 = Conv2D(64, (3, 3), padding='same', activation='relu')(branch4)
    concat_branch = Concatenate(axis=-1)([branch1, branch2, branch3, branch4])

    # Flatten and fully connected layers
    x = Flatten()(concat_branch)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(10, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Instantiate the model
model = dl_model()
model.summary()