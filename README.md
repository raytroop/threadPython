### 1. custom thread safe generator
`keras.utils.Sequence()` is more elegament and thread safer method than custom generator

[An exmaple](https://blog.csdn.net/u011311291/article/details/80991330)
```python
#coding=utf-8
'''
Created on 2018-7-10

'''
import keras
import math
import os
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class DataGenerator(keras.utils.Sequence):

    def __init__(self, datas, batch_size=1, shuffle=True):
        self.batch_size = batch_size
        self.datas = datas
        self.indexes = np.arange(len(self.datas))
        self.shuffle = shuffle

    def __len__(self):
        #计算每一个epoch的迭代次数
        return math.ceil(len(self.datas) / float(self.batch_size))

    def __getitem__(self, index):
        #生成每个batch数据，这里就根据自己对数据的读取方式进行发挥了
        # 生成batch_size个索引
        batch_indexs = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # 根据索引获取datas集合中的数据
        batch_datas = [self.datas[k] for k in batch_indexs]

        # 生成数据
        X, y = self.data_generation(batch_datas)

        return X, y

    def on_epoch_end(self):
        #在每一次epoch结束是否需要进行一次随机，重新随机一下index
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, batch_datas):
        images = []
        labels = []

        # 生成数据
        for i, data in enumerate(batch_datas):
            #x_train数据
            image = cv2.imread(data)
            image = list(image)
            images.append(image)
            #y_train数据
            right = data.rfind("\\",0)
            left = data.rfind("\\",0,right)+1
            class_name = data[left:right]
            if class_name=="dog":
                labels.append([0,1])
            else:
                labels.append([1,0])
        return np.array(images), np.array(labels)

# 读取样本名称，然后根据样本名称去读取数据
class_num = 0
train_datas = []
for file in os.listdir("D:/xxx"):
    file_path = os.path.join("D:/xxx", file)
    if os.path.isdir(file_path):
        class_num = class_num + 1
        for sub_file in os.listdir(file_path):
            train_datas.append(os.path.join(file_path, sub_file))

# 数据生成器
training_generator = DataGenerator(train_datas)

#构建网络
model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=784))
model.add(Dense(units=2, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(training_generator, epochs=50,max_queue_size=10,workers=1)
```

### 2. `ThreadPoolExecutor` in `concurrent.futures`
preprocess images efficiently by multithread

- 2.1 [fastai/dataset.py](https://github.com/fastai/fastai/blob/74cc80fd99df4874fcd1cfd5ce5db0f771106b3a/fastai/dataset.py#L74-L83)
```python
    def safely_process(fname):
        try:
            resize_img(fname, targ, path, new_path, fn=fn)
        except Exception as ex:
            errors[fname] = str(ex)


    if len(fnames) > 0:
        with ThreadPoolExecutor(num_cpus()) as e:
            ims = e.map(lambda fname: safely_process(fname), fnames)
            for _ in tqdm(ims, total=len(fnames), leave=False): pass
```
- 2.2 [dl2/carvana.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/carvana.ipynb)
```python
def resize_mask(fn):
    Image.open(fn).resize((128,128)).save((fn.parent.parent)/'train_masks-128'/fn.name)

files = list((PATH/'train_masks_png').iterdir())
with ThreadPoolExecutor(8) as e:
    e.map(resize_mask, files)
```

#### credits:
- [Kaggle-Carvana-3rd-Place-Solution](Kaggle-Carvana-3rd-Place-Solution/README.md)
- [https://blog.csdn.net/u011311291/article/details/80991330](https://blog.csdn.net/u011311291/article/details/80991330)
- [https://github.com/fastai/fastai/blob/master/courses/dl2/carvana.ipynb](https://github.com/fastai/fastai/blob/master/courses/dl2/carvana.ipynb)