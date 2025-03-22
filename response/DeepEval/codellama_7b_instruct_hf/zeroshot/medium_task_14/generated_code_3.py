from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, Dense, Flatten
from keras.applications.vgg16 import VGG16


def dl_model():
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))

    # Convolutional layers
    conv_block1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    conv_block2 = Conv2D(64, (3, 3), activation='relu')(conv_block1)
    conv_block3 = Conv2D(64, (3, 3), activation='relu')(conv_block2)

    # Batch normalization and ReLU activation
    batch_norm1 = BatchNormalization()(conv_block1)
    batch_norm2 = BatchNormalization()(conv_block2)
    batch_norm3 = BatchNormalization()(conv_block3)

    relu1 = ReLU()(batch_norm1)
    relu2 = ReLU()(batch_norm2)
    relu3 = ReLU()(batch_norm3)

    # Parallel branch
    parallel_branch = Conv2D(64, (3, 3), activation='relu')(input_layer)

    # Addition layer
    add_layer = Add()([relu1, relu2, relu3, parallel_branch])

    # Flatten layer
    flatten_layer = Flatten()(add_layer)

    # Dense layers
    dense_layer1 = Dense(128, activation='relu')(flatten_layer)
    dense_layer2 = Dense(64, activation='relu')(dense_layer1)

    # Output layer
    output_layer = Dense(10, activation='softmax')(dense_layer2)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model 