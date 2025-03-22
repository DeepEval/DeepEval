import keras
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Split input into three groups for main path
    split1 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    split2 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    split3 = Lambda(lambda x: keras.backend.split(x, num_or_size_splits=3, axis=-1))(input_layer)
    
    # Main path: three feature extraction paths with different kernel sizes
    def feature_extraction_path(split_tensor):
        # Convolutional layers with different kernel sizes
        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(split_tensor[0])
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(split_tensor[1])
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(split_tensor[2])
        # Concatenate the outputs
        concat = Concatenate()(list(map(lambda t: t, [conv1, conv2, conv3])))
        return concat
    
    main_output = feature_extraction_path(split1)
    main_output = feature_extraction_path(split2)
    main_output = feature_extraction_path(split3)
    main_output = Concatenate()([main_output, main_output, main_output])
    
    # Branch path: 1x1 convolution to align channels
    branch_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Combine main and branch paths
    fused_features = keras.backend.sum([main_output, branch_output], axis=0)
    
    # Classify using fully connected layers
    dense1 = Dense(units=512, activation='relu')(fused_features)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model