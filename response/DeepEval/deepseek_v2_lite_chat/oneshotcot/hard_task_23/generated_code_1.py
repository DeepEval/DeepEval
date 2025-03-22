import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, AveragePooling2D
from keras.models import Model

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32

    # Initial 1x1 convolutional layer
    conv1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(input_layer)
    
    # First branch for local feature extraction
    branch1 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(conv1)
    branch1 = BatchNormalization()(branch1)
    
    # Second branch for downsampling
    branch2 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch2)
    branch2 = BatchNormalization()(branch2)
    
    # Third branch for downsampling and upsampling
    branch3 = AveragePooling2D(pool_size=(2, 2))(input_layer)
    branch3 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(branch3)
    branch3 = BatchNormalization()(branch3)
    branch3 = Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(2, 2))(branch3)  # upsampling
    branch3 = BatchNormalization()(branch3)

    # Concatenate features from all branches
    concat = Concatenate()([branch1, branch2, branch3])
    
    # 1x1 convolutional layer to refine the concatenated features
    refined = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(concat)
    
    # Fully connected layer
    dense = Dense(units=512, activation='relu')(refined)
    
    # Output layer with 10 classes
    output_layer = Dense(units=10, activation='softmax')(dense)

    # Construct the model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model

# Create the model
model = dl_model()
model.summary()