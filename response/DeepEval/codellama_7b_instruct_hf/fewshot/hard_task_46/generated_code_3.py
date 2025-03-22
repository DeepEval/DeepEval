import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, Lambda
from keras.models import Model

def dl_model():
    # First block
    input_layer = Input(shape=(32, 32, 3))
    conv1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu')(input_layer)
    conv2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(conv1)
    conv3 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu')(conv2)
    output1 = Add()([conv1, conv2, conv3])
    
    # Second block
    branch1 = Conv2D(32, kernel_size=(1, 1), strides=(1, 1), activation='relu')(output1)
    branch2 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu')(output1)
    branch3 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu')(output1)
    branch4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(output1)
    output2 = Add()([branch1, branch2, branch3, branch4])
    
    # Global average pooling
    output3 = Flatten()(output2)
    output3 = Dense(512, activation='relu')(output3)
    output3 = Dense(10, activation='softmax')(output3)
    
    # Create and compile model
    model = Model(inputs=input_layer, outputs=output3)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model