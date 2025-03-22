# Import necessary packages
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Dropout, concatenate, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

# Define the function to create the deep learning model
def dl_model():
    # Define the input shape
    input_shape = (32, 32, 3)

    # Create the input layer
    inputs = Input(shape=input_shape)

    # Define branch 1
    branch1 = Conv2D(32, (1, 1), kernel_regularizer=l2(0.01))(inputs)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Conv2D(32, (3, 3), kernel_regularizer=l2(0.01))(branch1)
    branch1 = BatchNormalization()(branch1)
    branch1 = Activation('relu')(branch1)
    branch1 = Dropout(0.2)(branch1)

    # Define branch 2
    branch2 = Conv2D(32, (1, 1), kernel_regularizer=l2(0.01))(inputs)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(32, (1, 7), kernel_regularizer=l2(0.01))(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(32, (7, 1), kernel_regularizer=l2(0.01))(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Conv2D(32, (3, 3), kernel_regularizer=l2(0.01))(branch2)
    branch2 = BatchNormalization()(branch2)
    branch2 = Activation('relu')(branch2)
    branch2 = Dropout(0.2)(branch2)

    # Define branch 3
    branch3 = MaxPooling2D((3, 3))(inputs)

    # Concatenate the outputs of all branches
    concatenated = concatenate([branch1, branch2, branch3], axis=-1)

    # Apply Global Average Pooling
    x = GlobalAveragePooling2D()(concatenated)

    # Define the fully connected layers
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(10, activation='softmax')(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    return model