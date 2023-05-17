import time
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from keras.layers import Dropout

df = pd.read_csv('Regression_data_preprocessing.csv')
target = 'Rings'
y = df[target]
x = df.drop(target, axis =1)

# method_custom_metric 구현
def accuracy(y_true, y_pred):
    return 1 - tf.abs((y_true - y_pred) / y_true) 


# 유저로 부터 입력을 받아 검증 데이터 셋을 사용할 것인지, 표준화를 사용할 것인지 정함.
def create_best():
    print("[안내] 모델링을 시작합니다. (y or n)으로 진행해주세요")
    input_1 = input("[안내] 데이터를 표준화 하시겠습니까? : ")
    input_2 = input("[안내] 검증 데이터셋을 분리할까요? : ")

    # 표준화 진행 여부
    if input_1 == 'y':
        scaler = StandardScaler()
        X = scaler.fit_transform(x)
        print("[안내] 데이터 표준화를 진행했습니다.")
    else:
        X = x
        print("[안내] 데이터 표준화를 진행하지 않습니다.")

    # 검증 데이터 진행 여부
    if input_2 == 'y':
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        print("[안내] 검증 데이터를 추가로 분리했습니다.")

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("[안내] 검증 데이터를 분리하지 않았습니다.")

    # 최고 모델 사용자 친화적 구현
    start_time = time.time()
    print("[안내] 모델이 실행됩니다.")
    model = Sequential()
    model.add(Dense(64, activation='tanh', input_dim=x.shape[1]))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(8, activation='tanh'))
    model.add(Dense(4, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='sgd', metrics=[accuracy])
    early_stopping = EarlyStopping(monitor='accuracy', patience=5)

    if input_2 == 'y':
        model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])
        y_pred = model.predict(X)
    else:
        model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
        y_pred = model.predict(X)

    print("[안내] 최종 모델")
    if input_2 == 'y':
        print("[안내] train loss, accuracy")
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
        print("[안내] validation loss, accuracy")
        loss, acc = model.evaluate(X_val, y_val, verbose=2)
    else:
        print("[안내] train loss, accuracy")
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=2)
        print("[안내] test loss, accuracy")
        loss, acc = model.evaluate(X_test, y_test, verbose=2)

    end_time = time.time()

    execution_time = end_time - start_time
    print("[안내] 실행 시간 : {:.3f} seconds".format(execution_time))

    input_3 = input("[안내] 예측 샘플을 확인할까요? : ")
    if input_3 == 'y':
        print("[안내] 샘플 10개의 결과")
        new_y = y
        stacked_array = np.vstack((y_pred))
        new_df = pd.DataFrame(stacked_array)
        new_y = pd.DataFrame(new_y)
        new_y['pred'] = new_df[0]
        print(new_y.sample(10))
        print("[안내] 실행을 종료합니다.")

    else:
        print("[안내] 실행을 종료합니다.")



# 실행
create_best()
