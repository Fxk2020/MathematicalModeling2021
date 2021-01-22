### TensorFlow对于cifar数据集的处理

未使用框架之前loss图

![自己实现神经网络](C:\Users\26292\Desktop\美赛冲冲冲\2021Mathematical modeling\MathematicalModeling2021\TensorFlow框架学习\cifar10\自己实现神经网络.png)

准确率为：0.459

![image-20210122182121963](C:\Users\26292\AppData\Roaming\Typora\typora-user-images\image-20210122182121963.png)



### 一、使用TensorFlow框架

#### 1.导入数据

**一定要进行数据的预处理**

```python
fashion_mnist = keras.datasets.cifar10

(train_images, train_labels), (test_images, test_labels) = fashion_mnist2.load_data()
# 数据预处理 否则会发生激活函数的溢出
train_images = train_images / 255.0
test_images = test_images / 255.0
```

#### 2.构建模型

输入层：32x32x3

隐藏层有500个神经元，激活函数是sigmoid

输出层有10个类别

```python
# 2.构建模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(32, 32, 3)),
    keras.layers.Dense(500, activation='sigmoid'),
    keras.layers.Dense(10)
])
```

#### 3.编译模型

- *优化器* （optimizer）- 决定模型如何根据其看到的数据和自身的损失函数进行更新。
- 损失函数（loss）-使用库函数
- 指标（metrics）-用于反映模型的准确率

```python
# 3.编译模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

#### 4.训练模型

训练50轮

```python
model.fit(train_images, train_labels, epochs=50)
```

#### 5.验证在测试集上的准确率

```python
# 5.计算在测试集上的准确率
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
```

准确率到达了0.456，关键是时间大大的缩短了只有8分多钟，相较于自己写的运行了一个多小时，性能大大提高了。

![image-20210122204838563](C:\Users\26292\AppData\Roaming\Typora\typora-user-images\image-20210122204838563.png)

但是准确率并没有明显的提高。

### 二、使用卷积神经网络

#### 1.导入数据

```python
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# 将像素的值标准化至0到1的区间内。
train_images, test_images = train_images / 255.0, test_images / 255.0
```

#### 2.构建卷积神经网络模型

##### tf.keras.layers.Conv2D

2D卷积层（例如，图像上的空间卷积）。

```python
tf.keras.layers.Conv2D(
    filters, kernel_size, strides=(1, 1), padding='valid',
    data_format=None, dilation_rate=(1, 1), groups=1, activation=None,
    use_bias=True, kernel_initializer='glorot_uniform',
    bias_initializer='zeros', kernel_regularizer=None,
    bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
    bias_constraint=None, **kwargs
)
```

该层创建一个卷积内核，该卷积内核与该层输入进行卷积以产生输出张量。如果`use_bias`为True，则会创建一个偏差矢量并将其添加到输出中。最后，如果 `activation`不是`None`，那么它也将应用于输出。

当将此层用作模型的第一层时，请提供关键字参数`input_shape` （整数元组，不包括采样轴），例如，`input_shape=(128, 128, 3)`用于中的128x128 RGB图片`data_format="channels_last"`。

##### tf.keras.layers.MaxPool2D

```python 
tf.keras.layers.MaxPool2D(
    pool_size=(2, 2), strides=None, padding='valid', data_format=None,
    **kwargs
)
```

通过`pool_size`沿要素轴为每个尺寸定义的窗口上的最大值，对输入表示进行下采样。窗口`strides`在每个维度上移动一次。使用“有效”填充选项时的结果输出的形状（行或列数）为： `output_shape = (input_shape - pool_size + 1) / strides)`

使用“相同”填充选项时，结果输出形状为： `output_shape = input_shape / strides`

例如，对于stride =（1,1）和padding =“ valid”：



```python
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```

#### 添加Dense层

```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
```

#### 3.编译并训练模型

```python
model.compile(optimizer='adam',
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
```

#### 4.计算在测试集上的准确率

我们可以看到准确率变得更高达到了0.7074，所需的时间更短达到了4.6分钟

这是一个非常简易的卷积神经网络达到了较好的效果。

![image-20210122213209652](C:\Users\26292\AppData\Roaming\Typora\typora-user-images\image-20210122213209652.png)

#### 5.画出loss图和准确率图

history中储存着训练过程中的数据

```python
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.4, 1])
plt.legend(loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.ylim([0, 2])
plt.legend(loc='lower right')
plt.show()
```