import keras
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense
from keras.models import Model

def basic_block(x, filters):
    # Main path
    main_path = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    main_path = BatchNormalization()(main_path)
    main_path = ReLU()(main_path)
    
    # Branch path (identity)
    branch = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    branch = BatchNormalization()(branch)
    
    # Add main path and branch path
    output = Add()([main_path, branch])
    output = ReLU()(output)
    
    return output

def dl_model():
    input_layer = Input(shape=(32, 32, 3))  # CIFAR-10 images are 32x32 with 3 channels
    
    # Initial convolution to adjust input dimensions
    x = Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu')(input_layer)
    
    # First level of the residual structure
    x = basic_block(x, 16)
    
    # Second level of the residual structure
    x = basic_block(x, 16)
    x = basic_block(x, 16)
    
    # Third level of the residual structure
    global_branch = Conv2D(16, kernel_size=(3, 3), padding='same')(x)
    x = Add()([x, global_branch])
    
    # Average pooling and dense layer for classification
    x = AveragePooling2D(pool_size=(8, 8))(x)
    x = Flatten()(x)
    output_layer = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()