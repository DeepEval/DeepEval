import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform

def conv_block(input_tensor, filters, kernel_size):
    x = Conv2D(filters, kernel_size=kernel_size, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def block(input_tensor):
    main_path = conv_block(input_tensor, filters=96, kernel_size=(1, 1))
    main_path = conv_block(main_path, filters=96, kernel_size=(3, 3))
    main_path = Conv2D(filters=192, kernel_size=(1, 1), padding='same')(main_path)

    branch_path = Conv2D(filters=96, kernel_size=(1, 1), padding='same')(input_tensor)

    output_tensor = Add()([main_path, branch_path])
    output_tensor = Activation('relu')(output_tensor)
    return output_tensor

def feature_extractor(input_tensor):
    x = conv_block(input_tensor, filters=64, kernel_size=(3, 3))
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = block(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = block(x)
    return x

def dl_model():
    input_tensor = Input(shape=(32, 32, 3))
    features = feature_extractor(input_tensor)
    features = Lambda(lambda x: tf.split(x, num_or_size_splits=3, axis=3))(features)
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), padding='same')(features[0])
    conv1 = Dropout(0.5)(conv1)
    conv2 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(features[1])
    conv2 = Dropout(0.5)(conv2)
    conv3 = Conv2D(filters=16, kernel_size=(5, 5), padding='same')(features[2])
    conv3 = Dropout(0.5)(conv3)
    x = Concatenate()([conv1, conv2, conv3])
    x = Flatten()(x)
    x = Dense(units=10, activation='softmax')(x)

    model = Model(inputs=input_tensor, outputs=x)
    return model