<!-- PROJECT LOGO -->
<br />
<p align="center">
  <!-- <a href="https://github.com/catiaspsilva/README-template">
    <img src="images/gators.jpg" alt="Logo" width="150" height="150">
  </a> -->

  <h3 align="center">SHEAF DIFFUSION 
    <br> 
    <span style="font-weight: normal; font-size: 0.8em;">
    for edge prediction
    </span>
  </h3>

  <!-- <p align="center">

  <p align="center">
    A README template to jumpstart your projects!
    <br />
    <a href="https://github.com/catiaspsilva/README-template/blob/main/images/docs.txt"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="#usage">View Demo</a>
    ·
    <a href="https://github.com/catiaspsilva/README-template/issues">Report Bug</a>
    ·
    <a href="https://github.com/catiaspsilva/README-template/issues">Request Feature</a>
  </p> -->
</p>


## О проекте

Проект посвящён исследованию и реализации идей статьи Neural Sheaf Diffusion для задачи восстановления рёбер в графах.

Модель представляет собой вариацию графовых диффузионных сетей, где ключевую роль играют построение отображений на рёбрах и использование sheaf-лапласианов, обобщающих классические графовые операторы.

**Задача модели :** edge prediction (link prediction): по признакам узлов и частично наблюдаемой структуре графа восстановить отсутствующие рёбра. Для обучения используется модифицированная бинарная кросс-энтропия с балансировкой классов.

**Оттличие от классических GNN :** sheaf-based подход позволяет задать более богатую геометрию взаимодействий между узлами. Это делает модель более выразительной и потенциально улучшает результаты в задачах восстановления структуры графа, особенно при высокой гетерогенности вершин.

## Основные компоненты
* MapsBuilder — обучаемая подсеть, которая для каждой пары вершин строит линейное отображение в пространстве признаков, задавая локальную структуру графа.

* LaplacianBuilder — модуль, формирующий sheaf Laplacian на основе отображений и степеней вершин, нормализуя операторы диффузии.

* Diffusion Layer — слой, моделирующий распространение признаков по графу через sheaf-структуру. Каждая итерация включает применение построенного лапласиана и обновление эмбеддингов узлов.

* Decoders (MLP, DotProduct, Bilinear) — разные варианты декодеров для восстановления матрицы смежности и предсказания наличия рёбер.

## Установка и запуск

### Концигурация и параметры экспериментов

Передача параметров осуществляется через концигурационный файл `./config/config.yaml` с использованием `Hydra` и `OmegaConf`.

### Установка

1. Клонирование репозитория
   ```sh
   git clone https://github.com/catiaspsilva/README-template.git
   ```
2. Развертывание окружения
   ```sh
   make
   ```
   Развертывание в Google Colab
   ```sh
   make -f Makefile.colab
   ```
3. Запуск эксперименов
   ```sh
   python main.py
   ```

### Мультистарт

Конфигурацией проекта предусмотрена возможность установка расписания экспериментов, а так же использование `Optuna` для подбора оптимальных гиперпараметров.
Для этого необходимо в файле концигурации указать небходимые параметры для мультистарта и осуществить запуск.

```sh
python main.py ---multirun
```
