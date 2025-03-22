import keras
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, Flatten, Dense, ZeroPadding2D

def dl_model():
    # Load and prepare the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    input_layer = Input(shape=(28, 28, 1))
    
    # Define the sequential layers
    conv1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(64, (3, 3), activation='relu')(pool1)
    
    # Implement the block with four parallel paths
    def block(input_tensor):
        path1 = Conv2D(64, (1, 1), activation='relu')(input_tensor)
        path2 = Conv2D(64, (3, 3), activation='relu')(input_tensor)
        path3 = Conv2D(64, (5, 5), activation='relu')(input_tensor)
        path4 = MaxPooling2D(pool_size=(1, 1))(input_tensor)
        output_tensor = Concatenate()([path1, path2, path3, path4])
        return output_tensor
    
    block_output = block(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(block_output)
    pooled_output = Flatten()(pool2)
    
    # Apply two fully connected layers
    dense1 = Dense(128, activation='relu')(pooled_output)
    dense2 = Dense(64, activation='relu')(dense1)
    output_layer = Dense(10, activation='softmax')(dense2)
    
    # Create the model
    model = Model(inputs=input_layer, outputs=output_layer)
    
    return model