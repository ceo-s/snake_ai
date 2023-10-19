# Snake AI

## Навигация

### Среда выполнения
**[Snake game](https://github.com/ceo-s/snake_ai/blob/main/snake_v0_ai.py)**

### Тесты

1. DQN модель с бинарным вектором на входе - **[Notebook](https://github.com/ceo-s/snake_ai/blob/main/q_snake_keras.ipynb)**, **[train.py file](https://github.com/ceo-s/snake_ai/blob/main/q_snake_keras_train.py)**
  Слабый инпут, слабая модель, но рабочий агент.
  Первые попытки:
  ![plot1](https://github.com/ceo-s/snake_ai/blob/main/media/training_progress.png)
  Лучший результат:
  ![plot2](https://github.com/ceo-s/snake_ai/blob/main/media/training_progress_finale.png)

2. DQN модель с вектором с дистанцией на входе - **[Notebook](https://github.com/ceo-s/snake_ai/blob/main/q_snake_keras_distanced.ipynb)**, **[train.py file](https://github.com/ceo-s/snake_ai/blob/main/q_snake_keras_train_distanced.py)**
  Пришлось поковыряться с архитектурой, чтобы сделать нормировку инпута как следует. Но для этого алгоритма все равно не прошло.

3. DQN модель с изображением на входе - **[Notebook](https://github.com/ceo-s/snake_ai/blob/main/q_snake_keras_img.ipynb)**, **[train.py file](https://github.com/ceo-s/snake_ai/blob/main/q_snake_keras_train_img.py)**
  Очень сложный инпут для этого алгоритма. Сеть обучалась долго, но так и не сошлась.

4. Генетический алгоритм с изображением на входе и вектором с дистанцией - **[Notebook](https://github.com/ceo-s/snake_ai/blob/main/gen_algo.ipynb)**
  В целом работало неплохо, но очень непредсказуемо. Вектор с дистанцией быстрее и успешнее. Это единственный подход, в котором для этого входа сработала самая простая полносвязная сетка без нормировки.

5. A2C модель с бинарным вектором на входе - **[Notebook](https://github.com/ceo-s/snake_ai/blob/main/a2c_snake.ipynb)**
  Сложная модель, сложно подобрать параметры. В этот раз не получилось.

6. A2C модель с изображением на входе - **[Notebook](https://github.com/ceo-s/snake_ai/blob/main/a2c_snake_img.ipynb)**
  Сложная модель, сложно подобрать параметры, так ещё и инпут тяжёлый. Всё было против этой сетки. По правде сказать, я думаю, что она показала бы лучший результат если бы обучение пошло.

## Запустить проект

1. Clone repo with `git clone https://github.com/ceo-s/snake_ai.git`
2. Install dependencies via `pip install requirements.txt` or poetry
3. Follow along with Notebooks or configure and execute any of `*_train.py` scripts to start learning
