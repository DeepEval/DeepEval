import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Concatenate, Dense, Flatten

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main pathway
    conv1_1 = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3 = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_1 = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    main_path = Add()([conv1_1, conv1_3, conv3_1])
    main_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(main_path)
    
    # Branch pathway
    conv1_1_branch = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv1_3_branch = Conv2D(filters=32, kernel_size=(1, 3), strides=(1, 1), padding='same', activation='relu')(input_layer)
    conv3_1_branch = Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', activation='relu')(input_layer)
    branch_path = Add()([conv1_1_branch, conv1_3_branch, conv3_1_branch])
    branch_path = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), activation='relu')(branch_path)
    
    # Concatenate the outputs from both paths
    concat = Concatenate()([main_path, branch_path])
    
    # Final 1x1 convolution
    output = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(concat)
    
    # Model architecture
    model = keras.Model(inputs=input_layer, outputs=output)
    
    return model

# Create the model
model = dl_model()
model.summary()