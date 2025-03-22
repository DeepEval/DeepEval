import keras
from keras.datasets import cifar10
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate, BatchNormalization, Flatten, Dense, Dropout

def dl_model():
    # Load and preprocess CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Input layer
    input_layer = Input(shape=(32, 32, 3))
    
    # Four branches for feature extraction
    def branch(input_tensor, drop_prob):
        conv1 = Conv2D(filters=32, kernel_size=1, padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(conv1)
        conv3 = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu')(conv2)
        avg_pool = MaxPooling2D(pool_size=(2, 2))(conv3)
        drop = Dropout(drop_prob)(avg_pool)
        return drop
    
    # Apply branches
    drop_branch1 = branch(input_tensor=input_layer, drop_prob=0.1)
    drop_branch2 = branch(input_tensor=drop_branch1, drop_prob=0.2)
    drop_branch3 = branch(input_tensor=drop_branch2, drop_prob=0.3)
    drop_branch4 = branch(input_tensor=drop_branch3, drop_prob=0.4)
    
    # Concatenate the outputs of branches
    concat_layer = Concatenate()([drop_branch1, drop_branch2, drop_branch3, drop_branch4])
    
    # Flatten and pass through three fully connected layers
    flatten = Flatten()(concat_layer)
    dense1 = Dense(units=128, activation='relu')(flatten)
    dense2 = Dense(units=64, activation='relu')(dense1)
    output_layer = Dense(units=10, activation='softmax')(dense2)
    
    # Construct the model
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    
    return model

# Build and print the model
model = dl_model()
model.summary()