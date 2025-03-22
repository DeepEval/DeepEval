import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda, Dropout
from keras.applications import VGG16


def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Split input into three groups along the channel dimension
    groups = Lambda(lambda x: tf.split(x, 3, axis=-1))(input_layer)

    # Main pathway
    main_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(groups[0])
    main_pathway = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(main_pathway)
    main_pathway = MaxPooling2D(pool_size=(2, 2), strides=2)(main_pathway)
    main_pathway = Dropout(0.2)(main_pathway)

    # Branch pathway
    branch_pathway = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same')(groups[1])
    branch_pathway = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same')(branch_pathway)
    branch_pathway = MaxPooling2D(pool_size=(2, 2), strides=2)(branch_pathway)
    branch_pathway = Dropout(0.2)(branch_pathway)

    # Concat main and branch pathways
    x = Concatenate()([main_pathway, branch_pathway])

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layers
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = Dense(units=10, activation='softmax')(x)

    # Create model
    model = keras.Model(inputs=input_layer, outputs=x)

    return model