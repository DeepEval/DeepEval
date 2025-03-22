from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Concatenate, Dense

def dl_model():
    # Block 1
    input_layer = Input(shape=(28, 28, 1))
    path1 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    path2 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    path3 = Conv2D(32, (3, 3), activation='relu')(input_layer)
    path1 = MaxPooling2D(pool_size=(2, 2))(path1)
    path2 = MaxPooling2D(pool_size=(2, 2))(path2)
    path3 = MaxPooling2D(pool_size=(2, 2))(path3)
    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path3 = Flatten()(path3)
    concatenated = Concatenate()([path1, path2, path3])
    dropout = Dropout(0.5)(concatenated)
    
    # Block 2
    path1 = Conv2D(64, (3, 3), activation='relu')(dropout)
    path2 = Conv2D(64, (3, 3), activation='relu')(dropout)
    path3 = Conv2D(64, (3, 3), activation='relu')(dropout)
    path4 = AveragePooling2D(pool_size=(2, 2))(dropout)
    path1 = Flatten()(path1)
    path2 = Flatten()(path2)
    path3 = Flatten()(path3)
    path4 = Flatten()(path4)
    concatenated = Concatenate()([path1, path2, path3, path4])
    dropout = Dropout(0.5)(concatenated)
    output_layer = Dense(10, activation='softmax')(dropout)
    
    # Build and return the model
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model