import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, GlobalAveragePooling2D

def dl_model():
    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    input_layer = Input(shape=(32, 32, 3))
    
    # Main path
    def main_path(input_tensor):
        conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(conv2)
        pool = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)
        
        return pool
    
    main_pool = main_path(input_tensor=input_layer)
    
    # Branch path
    def branch_path(input_tensor):
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        dense1 = Dense(units=512, activation='relu')(avg_pool)
        dense2 = Dense(units=256, activation='relu')(dense1)
        
        return dense2
    
    branch_output = branch_path(input_tensor=main_pool)
    
    # Combine paths
    combined_output = keras.layers.concatenate([main_pool, branch_output])
    
    # Additional fully connected layers for classification
    dense3 = Dense(units=256, activation='relu')(combined_output)
    output_layer = Dense(units=10, activation='softmax')(dense3)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Instantiate and print the model
model = dl_model()
model.summary()