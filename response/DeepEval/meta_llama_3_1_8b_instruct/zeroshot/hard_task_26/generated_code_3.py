from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dense, GlobalAveragePooling2D
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers
from keras.optimizers import Adam

def dl_model():
    # Define the input shape for the CIFAR-10 dataset
    input_shape = (32, 32, 3)

    # Define the main input layer
    inputs = Input(shape=input_shape)

    # 1x1 initial convolutional layer
    x = Conv2D(16, (1, 1), padding='same', activation='relu')(inputs)

    # Define the main path
    main_path = Conv2D(16, (3, 3), padding='same', activation='relu')(x)

    # Define Branch 1
    branch_1 = Conv2D(32, (3, 3), padding='same', activation='relu')(main_path)

    # Define Branch 2
    branch_2 = MaxPooling2D((2, 2))(main_path)
    branch_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_2)
    branch_2 = UpSampling2D((2, 2))(branch_2)

    # Define Branch 3
    branch_3 = MaxPooling2D((2, 2))(main_path)
    branch_3 = Conv2D(32, (3, 3), padding='same', activation='relu')(branch_3)
    branch_3 = UpSampling2D((2, 2))(branch_3)

    # Concatenate the outputs from all branches
    x = concatenate([branch_1, branch_2, branch_3])

    # Add another 1x1 convolutional layer to produce the final output of the main path
    x = Conv2D(16, (1, 1), padding='same', activation='relu')(x)

    # Define the branch path
    branch_path = Conv2D(16, (1, 1), padding='same', activation='relu')(x)

    # Add the outputs from both paths
    x = add([x, branch_path])

    # Flatten the output
    x = GlobalAveragePooling2D()(x)

    # Add two fully connected layers for classification across 10 classes
    x = Dense(128, activation='relu')(x)
    outputs = Dense(10, activation='softmax')(x)

    # Define the model
    model = Model(inputs, outputs)

    # Compile the model
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    return model