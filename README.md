# ai_lab_2
# Лабораторная работа "Свёрточные нейронные сети"
Подоляка Е.О.
## Задание

1. Решить задачу классификации пород кошек и собак на основе датасета [Pet Faces][PetFaces]. 
![Dataset we will deal with](images/data.png)
2. Решить задачу классификации пород кошек и собак на основе датасета [Oxford Pets] с использованием Transfer Learning.
3. Обучить автоэнкодер или генеративно-состязательную сеть для генерации изображений лиц домашних животных. 
## 1. Pet Faces

Датасет [Pet Faces][PetFaces] представляет собой множество изображений 13 пород кошек и 23 пород собак, по 200 изображений на каждую породу. Изображение центрированы и уменьшены до небольшого размера. 

Вам необходимо обучить свёрточные нейронные сети для решения двух задач классификации:

* Определение кошки или собаки
* Определение породы кошки или собаки 

В обоих случаях вам необходимо самостоятельно придумать архитектуру сети, реализовать предобработку входных данных, разделить данные на обучающий и тестовый датасеты с сохранением пропорции по каждому из классов (stratified split).

Для загрузки датасета используйте следующий код:

```python
!wget http://www.soshnikov.com/permanent/data/petfaces.tar.gz
!tar xfz petfaces.tar.gz
!rm petfaces.tar.gz
```

В качестве результата необходимо:

* Посчитать точность классификатора на тестовом датасете
* Посчитать точность двоичной классификации "кошки против собак" на тестовом датасете
* Построить confusion matrix
* **[На хорошую и отличную оценку]** Посчитать top-3 accuracy
* **[На отличную оценку]** Выполнить оптимизацию гиперпараметров: архитектуры сети, learning rate, количества нейронов и размеров фильтров.

Решение оформите в файле [Faces.ipynb](Faces.ipynb).

## 2. Oxford Pets и Transfer Learing

Решить задачу классификации пород кошек и собак на основе датасета [Oxford-IIIT](https://www.robots.ox.ac.uk/~vgg/data/pets/).

![Dataset we will deal with](images/data.png)

Используйте оригинальный датасет **[Oxford Pets](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset)** и предобученные сети VGG-16/VGG-19 и ResNet для построения классификатора пород. 

В качестве результата необходимо:

* Обучить три классификатора пород: на основе VGG-16/19 и на основе ResNet.
* Посчитать точность классификатора на тестовом датасете отдельно для каждого из классификаторов, для дальнейших действий выбрать сеть с лучшей точностью
* Посчитать точность двоичной классификации "кошки против собак" такой сетью на тестовом датасете
* Построить confusion matrix
* **[На отличную оценку]** Посчитать top-3 и top-5 accuracy

Решение оформите в файле [Pets.ipynb](Pets.ipynb).

## 3. Генерация изображений

Натренируйте генеративную модель - автоэнкодер или генеративно-состязательную сеть для генерации изображений животных. Используйте один из датасетов на выбор: Pet Faces или полноценный Oxford Pets. Рекомендуется начинать экспериментировать с изображениями небольшого размера - 64x64 пикселя, и затем посмотреть, получится ли увеличить размер до 128x128 или 256x256.

Это задание сложнее предыдущих, поэтому делайте его все вместе!

[PetFaces]: https://www.soshnikov.com/permanent/data/petfaces.tar.gz
