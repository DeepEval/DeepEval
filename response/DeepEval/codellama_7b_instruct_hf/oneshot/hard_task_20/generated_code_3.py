import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16

def dl_model():
    # Load the VGG16 model
    vgg_model = VGG16(include_top=False, input_shape=(32, 32, 3))

    # Freeze the VGG16 model
    for layer in vgg_model.layers:
        layer.trainable = False

    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))

    # Define the main path
    main_path = vgg_model.get_layer('block1_conv1').output
    main_path = Conv2D(64, kernel_size=(1, 1), strides=(1, 1), padding='same')(main_path)
    main_path = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(main_path)
    main_path = Conv2D(256, kernel_size=(5, 5), strides=(1, 1), padding='same')(main_path)
    main_path = Concatenate()([main_path, main_path, main_path])
    main_path = BatchNormalization()(main_path)
    main_path = Flatten()(main_path)
    main_path = Dense(512, activation='relu')(main_path)
    main_path = Dense(10, activation='softmax')(main_path)

    # Define the branch path
    branch_path = Conv2D(128, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    branch_path = BatchNormalization()(branch_path)
    branch_path = Flatten()(branch_path)
    branch_path = Dense(512, activation='relu')(branch_path)
    branch_path = Dense(10, activation='softmax')(branch_path)

    # Define the fused features
    fused_features = Concatenate()([main_path, branch_path])
    fused_features = BatchNormalization()(fused_features)
    fused_features = Flatten()(fused_features)
    fused_features = Dense(512, activation='relu')(fused_features)
    fused_features = Dense(10, activation='softmax')(fused_features)

    # Define the model
    model = Model(inputs=input_layer, outputs=fused_features)

    return model