import keras
from keras.layers import Input, Conv2D, AveragePooling2D, UpSampling2D, Add, Flatten, Dense
from keras.layers import Concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))
    
    # Main Path
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    split1 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(conv1)
    
    branch1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split1[0])
    branch2 = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(conv1)
    branch2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(3, 3))(branch2)
    
    branch3 = AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')(conv1)
    branch3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(5, 5))(branch3)
    
    adding_layer1 = Add()([branch1, branch2, branch3])
    
    conv2 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(adding_layer1)
    split2 = Lambda(lambda x: tf.split(value=x, num_or_size_splits=3, axis=-1))(conv2)
    
    branch4 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split2[0])
    branch5 = AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')(adding_layer1)
    branch5 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch5)
    branch5 = UpSampling2D(size=(3, 3))(branch5)
    
    branch6 = AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')(adding_layer1)
    branch6 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch6)
    branch6 = UpSampling2D(size=(5, 5))(branch6)
    
    adding_layer2 = Add()([branch4, branch5, branch6])
    
    main_path_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(adding_layer2)
    
    # Branch Path
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Fuse Main Path and Branch Path Outputs
    fuse_layer = Add()([main_path_output, branch_path])
    
    # Fully Connected Layer
    flatten = Flatten()(fuse_layer)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model