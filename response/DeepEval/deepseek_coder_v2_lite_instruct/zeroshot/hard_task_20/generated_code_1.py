import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Define input shape
    input_shape = (32, 32, 3)
    
    # Define the input layer
    inputs = Input(shape=input_shape)
    
    # Main path
    main_path = Conv2D(32, (1, 1), activation='relu')(inputs)
    main_path = Conv2D(32, (3, 3), activation='relu', padding='same')(main_path)
    main_path = Conv2D(32, (5, 5), activation='relu', padding='same')(main_path)
    
    # Split the main path into three groups
    split_1 = Lambda(lambda x: x[:, :, :, :x.shape[3]//3])(main_path)
    split_2 = Lambda(lambda x: x[:, :, :, x.shape[3]//3:2*x.shape[3]//3])(main_path)
    split_3 = Lambda(lambda x: x[:, :, :, 2*x.shape[3]//3:])(main_path)
    
    # Concatenate the outputs of the three groups
    concatenated = tf.concat([split_1, split_2, split_3], axis=-1)
    
    # Branch path
    branch_path = Conv2D(32, (1, 1), activation='relu')(inputs)
    
    # Add the outputs of the main and branch paths
    fused_features = Add()([concatenated, branch_path])
    
    # Flatten the fused features
    flattened = Flatten()(fused_features)
    
    # Add fully connected layers
    fc1 = Dense(128, activation='relu')(flattened)
    fc2 = Dense(10, activation='softmax')(fc1)
    
    # Define the model
    model = Model(inputs=inputs, outputs=fc2)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()