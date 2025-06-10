import tensorflow
from keras.models import Sequential
from keras.layers import Dense, Input, Flatten
from keras.utils import to_categorical, set_random_seed
from keras.backend import clear_session
import numpy as np
import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
from keras.models import load_model
from PIL import Image

if 'global_var' not in st.session_state:
    st.session_state.global_var = False

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
X_train = X_train / 255
X_test = X_test / 255
y_train_ohe = to_categorical(y_train, num_classes = 10)
y_test_ohe = to_categorical(y_test, num_classes = 10)

def pred(img1):
    model = load_model('model.keras')
    img_pil = Image.open(img1)
    img = img_pil.copy()
    img = np.array(img.convert('L').resize((28,28)))
    img = img.reshape(1,28,28)
    img = 255 - img
    img = img /255
    pred = model.predict(img)
    pred_id = np.argsort(pred[0])[-3:][::-1]
    d = [' '.join([np.array(labels)[i], str(round(pred[0][i]*100, 2))]) for i in pred_id]
    st.text('\n'.join(d))

def create(k):
    clear_session()
    set_random_seed(10)
    model = Sequential()
    model.add(Input(shape = (X_train.shape[1], X_train.shape[2])))
    model.add(Flatten())
    model.add(Dense(10, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    model.summary()
    
    history = model.fit(X_train, y_train_ohe, epochs = k,verbose = 0)
    model.save('model.keras')
    return history

def visualize_data():
    fig, axs = plt.subplots(10, 10)
    fig.set_figheight(8)
    fig.set_figwidth(8)

    for i in range(10):
      ids = np.where(y_train == i)[0]
      for j in range(10):
        target = np.random.choice(ids)
        axs[i][j].axis('off')
        axs[i][j].imshow(X_train[target].reshape(28,28), cmap='gray')
    st.pyplot(fig)

def visualize():
    st.title('MINIST-Fashion Classification')
    t1, t2 = st.tabs(['Train', 'Inference'])
    with t1:
        col1,col2 = st.columns(2)
        with col1:
            st.text('Dataset')
            visualize_data()
        with col2:
            ep = st.number_input('Epochs', min_value = 1, value = 5)
            if st.button('Train', use_container_width= True):
                st.session_state.global_var = True
                with st.spinner('Training'): 
                    history = create(ep)
                with st.spinner('Evaluating'):
                    model = load_model('model.keras') 
                    _, acc = model.evaluate(X_test, y_test_ohe)
                    st.info(f'Model trained and saved. Test accuracy {round(acc*100)}%')
                    fig,ax = plt.subplots()
                    ax.set_title('Learning Curve')
                    ax.plot(history.history['accuracy'])
                    ax.plot(history.history['loss'])
                    ax.set_xlabel('Epochs')
                    ax.set_ylabel('Loss | Accuracy')
                    ax.legend(['Accuracy', 'Loss'])
                    st.pyplot(fig)
            else: st.session_state.global_var = False
    with t2:
        img = st.file_uploader('Uploade Image File', type = ['PNG', 'JPG', 'JPEG'])
        if img: print(type(img))
        c1,c2 = st.columns(2)
        if img:
            with c1:
                st.image(img)
            with c2:
                if st.session_state.global_var:
                    st.header('Prediction')
                    pred(img)
                else: st.error('Model not found') 
                    
                
        
visualize()