import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Conv2DTranspose, Add, Concatenate, Flatten, Dense

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    main_conv1x1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    
    # Branch 1 of Main Path
    main_branch1_conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_conv1x1)

    # Branch 2 of Main Path
    main_branch2_avgpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv1x1)
    main_branch2_conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_branch2_avgpool)
    main_branch2_upsample = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(main_branch2_conv3x3)

    # Branch 3 of Main Path
    main_branch3_avgpool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(main_conv1x1)
    main_branch3_conv3x3 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(main_branch3_avgpool)
    main_branch3_upsample = Conv2DTranspose(filters=32, kernel_size=(2, 2), strides=(2, 2), padding='same', activation='relu')(main_branch3_conv3x3)

    # Concatenate branches
    main_concatenated = Concatenate()([main_branch1_conv3x3, main_branch2_upsample, main_branch3_upsample])

    # Final 1x1 Conv in Main Path
    main_output = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(main_concatenated)

    # Branch Path
    branch_conv1x1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)

    # Fuse Main Path and Branch Path
    fused_output = Add()([main_output, branch_conv1x1])

    # Fully connected layer for classification
    flatten_layer = Flatten()(fused_output)
    output_layer = Dense(units=10, activation='softmax')(flatten_layer)

    # Create Model
    model = keras.Model(inputs=input_layer, outputs=output_layer)

    return model