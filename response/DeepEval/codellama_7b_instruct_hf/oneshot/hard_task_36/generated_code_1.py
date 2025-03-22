import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Add, GlobalAveragePooling2D

def dl_model():
    input_layer = Input(shape=(28, 28, 1))

    # Main pathway
    main_pathway = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_pathway = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_pathway)
    main_pathway = MaxPooling2D(pool_size=(2, 2), strides=2)(main_pathway)
    main_pathway = Dropout(0.5)(main_pathway)

    # Branch pathway
    branch_pathway = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse the two pathways
    x = Add()([main_pathway, branch_pathway])
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    x = Dense(units=64, activation='relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model