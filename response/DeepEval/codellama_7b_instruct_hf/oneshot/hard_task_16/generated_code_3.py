import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense
from keras.applications import VGG16

def dl_model():
    # Load the VGG-16 model as the main path
    base_model = VGG16(include_top=False, input_shape=(32, 32, 3))
    model = base_model

    # Block 1: Split the input into three groups and extract deep features
    input_layer = Input(shape=(32, 32, 3))
    x = input_layer
    x1 = Lambda(lambda x: x[:, :, :, :int(x.shape[-1] / 3)])(x)
    x2 = Lambda(lambda x: x[:, :, :, int(x.shape[-1] / 3):int(x.shape[-1] / 3) * 2])(x)
    x3 = Lambda(lambda x: x[:, :, :, int(x.shape[-1] / 3) * 2:])(x)
    x = Concatenate()([x1, x2, x3])
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = Conv2D(64, (1, 1), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    model.add(x)

    # Block 2: Global max pooling and channel-matching weights
    x = model.output
    x = GlobalMaxPooling2D()(x)
    x = Dense(x.shape[-1], activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    model.add(x)

    # Add the branch to the input and concatenate with the main path
    input_layer = Input(shape=(32, 32, 3))
    x = Concatenate()([input_layer, model.output])

    # Final layer for classification
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)
    model.add(x)

    return model