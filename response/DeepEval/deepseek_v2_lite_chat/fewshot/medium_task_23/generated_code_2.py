import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Flatten, Dense

def dl_model():
    # Define the input shape
    input_shape = (224, 224, 3)  # CIFAR-10 images are 32x32, but we adjust for padding

    # Path 1: Single 1x1 convolution
    path1_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(Input(shape=input_shape))
    
    # Path 2: 1x1 convolution followed by 1x7 and 7x1 convolutions
    path2_1 = Conv2D(filters=32, kernel_size=(1, 1), activation='relu')(path1_1)
    path2_2 = Conv2D(filters=32, kernel_size=(1, 7), padding='same')(path1_1)
    path2_3 = Conv2D(filters=32, kernel_size=(7, 1), padding='same')(path1_1)
    
    # Path 3: 1x1 convolution followed by two sets of 1x7 and 7x1 convolutions
    path3_1 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path1_1)
    path3_2 = concatenate([Conv2D(filters=64, kernel_size=(1, 7), padding='same')(path1_1),
                           Conv2D(filters=64, kernel_size=(7, 1), padding='same')(path1_1)])
    
    # Path 4: Average pooling followed by 1x1 convolution
    path4_1 = MaxPooling2D(pool_size=(2, 2))(Input(shape=input_shape))
    path4_2 = Conv2D(filters=64, kernel_size=(1, 1), activation='relu')(path4_1)
    
    # Concatenate the outputs of all paths
    concat_layer = concatenate([path2_1, path2_2, path2_3, path3_2, path4_2])
    
    # Fully connected layer and classification head
    output_layer = Dense(units=10, activation='softmax')(Flatten()(concat_layer))
    
    # Build the model
    model = Model(inputs=Input(shape=input_shape), outputs=output_layer)
    
    return model

# Instantiate the model
model = dl_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])