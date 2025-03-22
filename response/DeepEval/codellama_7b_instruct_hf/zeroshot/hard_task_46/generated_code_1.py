import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation, concatenate, GlobalAveragePooling2D, Flatten, Dense
from keras.models import Model


def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Define the first block
    x = Input(shape=input_shape)
    x1 = Lambda(lambda x: tf.split(x, 3, axis=1))(x)
    x1 = Conv2D(64, (1, 1), activation='relu', padding='same', name='block1_conv1')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x1)
    x1 = Conv2D(64, (5, 5), activation='relu', padding='same', name='block1_conv3')(x1)
    x1 = concatenate([x1], axis=1)

    # Define the second block
    x2 = Lambda(lambda x: tf.split(x, 3, axis=1))(x)
    x2 = Conv2D(64, (1, 1), activation='relu', padding='same', name='block2_conv1')(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block2_conv2')(x2)
    x2 = Conv2D(64, (5, 5), activation='relu', padding='same', name='block2_conv3')(x2)
    x2 = concatenate([x2], axis=1)

    # Define the third block
    x3 = Lambda(lambda x: tf.split(x, 3, axis=1))(x)
    x3 = Conv2D(64, (1, 1), activation='relu', padding='same', name='block3_conv1')(x3)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block3_conv2')(x3)
    x3 = Conv2D(64, (5, 5), activation='relu', padding='same', name='block3_conv3')(x3)
    x3 = concatenate([x3], axis=1)

    # Define the fourth block
    x4 = Lambda(lambda x: tf.split(x, 3, axis=1))(x)
    x4 = Conv2D(64, (1, 1), activation='relu', padding='same', name='block4_conv1')(x4)
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block4_conv2')(x4)
    x4 = Conv2D(64, (5, 5), activation='relu', padding='same', name='block4_conv3')(x4)
    x4 = concatenate([x4], axis=1)

    # Define the final layers
    x = concatenate([x1, x2, x3, x4], axis=1)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=x, outputs=x)

    # Compile the model
    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    return model