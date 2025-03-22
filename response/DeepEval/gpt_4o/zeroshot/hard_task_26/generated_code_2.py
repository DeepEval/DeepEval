from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, Add, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    # Initial 1x1 Convolution
    main_conv1x1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Branch 1: 3x3 Convolution
    branch1 = Conv2D(32, (3, 3), padding='same', activation='relu')(main_conv1x1)

    # Branch 2: Max Pool -> 3x3 Conv -> Upsample
    branch2 = MaxPooling2D(pool_size=(2, 2))(main_conv1x1)
    branch2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch2)
    branch2 = UpSampling2D(size=(2, 2))(branch2)

    # Branch 3: Max Pool -> 3x3 Conv -> Upsample
    branch3 = MaxPooling2D(pool_size=(2, 2))(main_conv1x1)
    branch3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch3)
    branch3 = UpSampling2D(size=(2, 2))(branch3)

    # Concatenate all branches
    concatenated_branches = Concatenate()([branch1, branch2, branch3])

    # Final 1x1 Convolution on the concatenated branches
    main_output = Conv2D(32, (1, 1), activation='relu')(concatenated_branches)

    # Branch path: 1x1 Convolution
    branch_conv1x1 = Conv2D(32, (1, 1), activation='relu')(input_layer)

    # Merge main and branch paths
    merged_output = Add()([main_output, branch_conv1x1])

    # Fully connected layers
    flatten = Flatten()(merged_output)
    fc1 = Dense(128, activation='relu')(flatten)
    fc2 = Dense(64, activation='relu')(fc1)

    # Output layer for 10 classes
    output_layer = Dense(10, activation='softmax')(fc2)

    # Construct model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Example usage
model = dl_model()
model.summary()