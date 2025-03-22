import keras
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Basic Block (Main Path)
    def block_main(x):
        # Convolution layer
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        # Batch normalization
        bn = BatchNormalization()(conv)
        # ReLU activation
        act = ReLU()(bn)
        return act
    
    # Basic Block (Branch Path)
    def block_branch(x):
        # Convolution layer
        conv = Conv2D(filters=32, kernel_size=(3, 3), padding='same')(x)
        # Average pooling
        avg_pool = AveragePooling2D(pool_size=(2, 2))(conv)
        # ReLU activation
        act = ReLU()(avg_pool)
        return act
    
    # Construct main and branch paths
    main_output = block_main(input_layer)
    branch_output = block_branch(input_layer)
    
    # Feature fusion
    fused_output = Add()([main_output, branch_output])
    
    # Downsampling layer
    avg_pool = AveragePooling2D(pool_size=(4, 4), strides=(4, 4))(fused_output)
    
    # Flatten and fully connected layer
    flatten = Flatten()(avg_pool)
    dense = Dense(units=512, activation='relu')(flatten)
    
    # Output layer
    output_layer = Dense(units=10, activation='softmax')(dense)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build the model
model = dl_model()
model.summary()