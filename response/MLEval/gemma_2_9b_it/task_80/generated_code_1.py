from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

def method():
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2)) 
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))  
    model.add(Dense(10, activation='softmax'))
    
    return model

# Validate by calling the method
model = method()
print(model.summary())