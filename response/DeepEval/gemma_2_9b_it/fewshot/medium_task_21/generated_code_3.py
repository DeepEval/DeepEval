import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Dropout, Flatten, Dense, Concatenate

def dl_model():
    
    input_layer = Input(shape=(32, 32, 3))

    # Branch 1: 1x1 convolution
    branch1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch1 = Dropout(0.25)(branch1)

    # Branch 2: 1x1 convolution followed by 3x3 convolution
    branch2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch2 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = Dropout(0.25)(branch2)

    # Branch 3: 1x1 convolution followed by two consecutive 3x3 convolutions
    branch3 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(input_layer)
    branch3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Conv2D(filters=256, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = Dropout(0.25)(branch3)

    # Branch 4: Average pooling followed by 1x1 convolution
    branch4 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch4 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(branch4)
    branch4 = Dropout(0.25)(branch4)

    # Concatenate outputs from all branches
    merged_features = Concatenate()( [branch1, branch2, branch3, branch4] )

    # Flatten and pass through fully connected layers
    flatten_layer = Flatten()(merged_features)
    dense1 = Dense(units=512, activation='relu')(flatten_layer)
    dense2 = Dense(units=256, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model