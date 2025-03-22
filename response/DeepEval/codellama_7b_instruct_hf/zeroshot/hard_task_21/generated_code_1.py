from keras.layers import Input, Lambda, Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Main path
    main_input = Input(shape=(32, 32, 3))
    x = main_input
    x = Lambda(lambda x: tf.split(x, 3, axis=3))(x)
    x = [Conv2D(32, (1, 1), strides=(1, 1), padding='same')(x[0]),
         Conv2D(64, (3, 3), strides=(1, 1), padding='same')(x[1]),
         Conv2D(128, (5, 5), strides=(1, 1), padding='same')(x[2])]
    x = tf.concat(x, axis=3)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Branch path
    branch_input = Input(shape=(32, 32, 3))
    x_branch = branch_input
    x_branch = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(x_branch)
    x_branch = BatchNormalization()(x_branch)
    x_branch = Activation('relu')(x_branch)

    # Add the outputs from both paths
    x = tf.add(x, x_branch)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Flatten the output and add fully connected layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(inputs=[main_input, branch_input], outputs=x)
    return model