import keras
from keras.datasets import mnist
from keras.layers import Input, Conv2D, Add, MaxPooling2D, Flatten, Dense, concatenate
from keras.models import Model

def dl_model():
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    input_A = Input(shape=(28, 28, 1))
    input_B = Input(shape=(28, 28, 1))
    
    # Block 1
    x = Conv2D(32, (3, 3), activation='relu')(input_A)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    branch_output_1 = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    x = Conv2D(64, (3, 3), activation='relu')(input_B)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    branch_output_2 = Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    combined = Add()([branch_output_1, branch_output_2])
    
    # Block 2
    x = MaxPooling2D(pool_size=(1, 1))(combined)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = MaxPooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    
    x = concatenate([x, branch_output_1])
    x = concatenate([x, branch_output_2])
    
    output = Dense(10, activation='softmax')(x)
    
    model = Model(inputs=[input_A, input_B], outputs=output)
    
    return model

# Create and compile the model
model = dl_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])