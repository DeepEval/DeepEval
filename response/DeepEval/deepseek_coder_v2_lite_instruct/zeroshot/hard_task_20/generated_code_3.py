import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Lambda, Add, Flatten, Dense

def dl_model():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize input data
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    
    # Define the input layer
    inputs = Input(shape=(32, 32, 3))
    
    # Main path
    main_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    main_path = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu')(main_path)
    main_path = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(main_path)
    
    # Split the main path into three groups
    split1 = Lambda(lambda x: x[:, :, :, :x.shape[3]//3])(main_path)
    split2 = Lambda(lambda x: x[:, :, :, x.shape[3]//3:2*x.shape[3]//3])(main_path)
    split3 = Lambda(lambda x: x[:, :, :, 2*x.shape[3]//3:])(main_path)
    
    # Concatenate the outputs of the three groups
    concatenated = tf.concat([split1, split2, split3], axis=-1)
    
    # Branch path
    branch_path = Conv2D(filters=32, kernel_size=(1, 1), padding='same', activation='relu')(inputs)
    
    # Add the main and branch paths
    fused_features = Add()([concatenated, branch_path])
    
    # Flatten the fused features
    flattened = Flatten()(fused_features)
    
    # Add fully connected layers for classification
    outputs = Dense(units=10, activation='softmax')(flattened)
    
    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Example usage
model = dl_model()
model.summary()