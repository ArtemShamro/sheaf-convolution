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

  <p align="center">

  <p align="center">
    A README template to jumpstart your projects!
    <!-- <br />
    <a href="https://github.com/catiaspsilva/README-template/blob/main/images/docs.txt"><strong>Explore the docs »</strong></a>
    <br /> -->
    <br />
    <a href="#usage">View Demo</a>
    ·
    <a href="https://github.com/catiaspsilva/README-template/issues">Report Bug</a>
    ·
    <a href="https://github.com/catiaspsilva/README-template/issues">Request Feature</a>
  </p>
</p>

<!-- ABOUT THE PROJECT -->

## О проекте

<!-- GETTING STARTED -->

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
