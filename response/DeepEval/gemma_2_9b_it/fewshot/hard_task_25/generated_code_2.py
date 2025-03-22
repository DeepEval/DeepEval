import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Add, Flatten, Dense

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    x = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(x)
    pool1 = MaxPooling2D(pool_size=(2, 2))(x)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(pool1)
    upsample1 = UpSampling2D(size=(2, 2))(branch2)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(upsample1)
    
    # Concatenate branches
    main_path = Add()([branch1, branch3])

    # Branch path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)

    # Fuse paths
    output_layer = Add()([main_path, branch_path])
    output_layer = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(output_layer)
    output_layer = Flatten()(output_layer)
    output_layer = Dense(units=10, activation='softmax')(output_layer)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model