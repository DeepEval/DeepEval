import keras
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Add, Flatten, Dense, BatchNormalization, ReLU, Concatenate

def dl_model():
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation=ReLU())(input_tensor)
        batch_norm1 = BatchNormalization()(conv1)
        main_pool1 = MaxPooling2D(pool_size=(2, 2))(batch_norm1)
        
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation=ReLU())(main_pool1)
        batch_norm2 = BatchNormalization()(conv2)
        main_pool2 = MaxPooling2D(pool_size=(2, 2))(batch_norm2)
        
        return batch_norm2
    
    # Branch path
    def branch_path(input_tensor):
        conv3 = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation=ReLU())(input_tensor)
        batch_norm3 = BatchNormalization()(conv3)
        
        return batch_norm3
    
    main_block = main_path(input_layer)
    branch_block = branch_path(input_layer)
    
    # Feature fusion
    fused_output = Add()([main_block, branch_block])
    
    # Classification layer
    flattened = Flatten()(fused_output)
    dense1 = Dense(units=128, activation='relu')(flattened)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Model construction
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Create the model
model = dl_model()
model.summary()