import keras
from keras.layers import Input, Conv2D, AveragePooling2D, Concatenate, BatchNormalization, Add, Activation, Conv2DTranspose, Dense
from keras import regularizers
from keras.optimizers import Adam

def dl_model():
    input_layer = Input(shape=(32, 32, 3))

    # Main Path
    conv1 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1 = BatchNormalization()(conv1)
    
    main_path = conv1
    branch1 = conv1
    branch2 = conv1
    branch3 = conv1

    # Branch 1: Local Features
    branch1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch1)
    branch1 = BatchNormalization()(branch1)

    # Branch 2: Downsample and Process
    branch2 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch2 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch2)

    # Branch 3: Downsample, Process and Upsample
    branch3 = AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(conv1)
    branch3 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Conv2DTranspose(filters=16, kernel_size=(2, 2), strides=(2, 2), padding='same')(branch3)

    # Concatenate the outputs of all branches
    output_tensor = Concatenate()([conv1, branch1, branch2, branch3])
    
    # Apply a 1x1 convolutional layer to form the main path output
    output_tensor = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(output_tensor)
    output_tensor = BatchNormalization()(output_tensor)
    
    # Branch Path
    branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(conv1)
    branch_path = BatchNormalization()(branch_path)

    # Fuse the main path and branch path outputs through addition
    output_tensor = Add()([output_tensor, branch_path])

    # Flatten the output
    output_tensor = Flatten()(output_tensor)
    
    # Fully connected layer for 10-class classification
    output_layer = Dense(units=10, activation='softmax')(output_tensor)

    model = keras.Model(inputs=input_layer, outputs=output_layer)

    # Compile the model
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    return model