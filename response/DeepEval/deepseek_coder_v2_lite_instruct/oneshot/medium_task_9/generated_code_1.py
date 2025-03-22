import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def dl_model():
    # Define the input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Initial convolutional layer to reduce dimensionality
    conv1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same')(input_layer)
    bn1 = BatchNormalization()(conv1)
    relu1 = ReLU()(bn1)
    
    # Main path of the first basic block
    main_path = relu1
    
    # Branch path of the first basic block
    branch_path = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same')(input_layer)
    
    # Feature fusion by adding the main path and branch path
    added = Add()([main_path, branch_path])
    
    # Second basic block
    conv2 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')(added)
    bn2 = BatchNormalization()(conv2)
    relu2 = ReLU()(bn2)
    
    main_path = relu2
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same')(added)
    added = Add()([main_path, branch_path])
    
    # Average pooling layer to downsample the feature map
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=1)(added)
    
    # Flatten the feature map
    flatten_layer = Flatten()(avg_pool)
    
    # Fully connected layer to produce the final classification output
    dense_layer = Dense(units=10, activation='softmax')(flatten_layer)
    
    # Construct the model
    model = Model(inputs=input_layer, outputs=dense_layer)
    
    return model