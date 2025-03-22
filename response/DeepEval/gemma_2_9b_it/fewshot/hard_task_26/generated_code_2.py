import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    x1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x1)
    branch2 = MaxPooling2D(pool_size=(2, 2))(x1)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    branch3 = MaxPooling2D(pool_size=(2, 2))(x1)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    main_path = concatenate([branch1, branch2, branch3], axis=3)
    main_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(main_path)

    # Branch Path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)

    # Add and Fully Connected Layers
    x = Add()([main_path, branch_path])
    x = Flatten()(x)
    x = Dense(units=128, activation='relu')(x)
    output_layer = Dense(units=10, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model