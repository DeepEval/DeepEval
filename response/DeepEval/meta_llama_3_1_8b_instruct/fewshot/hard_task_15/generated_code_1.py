import keras
from keras.layers import Input, Conv2D, AveragePooling2D, GlobalAveragePooling2D, Reshape, Multiply, Add, Flatten, Dense

def dl_model():

    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    pool1 = GlobalAveragePooling2D()(conv1)
    dense1 = Dense(units=64, activation='relu')(pool1)
    weights = Dense(units=64, activation='linear')(dense1)
    weights = Reshape(target_shape=(1, 1, 64))(weights)
    weighted_conv1 = Multiply()([conv1, weights])
    pool2 = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(weighted_conv1)
    flatten1 = Flatten()(pool2)

    # Branch Path
    branch_conv = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch_conv)
    flatten2 = Flatten()(branch_pool)

    # Combine Main and Branch Paths
    adding_layer = Add()([flatten1, flatten2])

    # Fully Connected Layers for Classification
    dense2 = Dense(units=128, activation='relu')(adding_layer)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model

model = dl_model()
print(model.summary())