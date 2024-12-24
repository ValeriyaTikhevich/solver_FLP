# Solver FLP (Facility Location Problem Solver) 🏢📍

**Описание**: Этот проект представляет собой решение задачи оптимизации расположения городских сервисов (Facility Location Problem, FLP) с использованием генетического алгоритма и метода линейного программирования. Задача заключается в нахождении наилучшего расположения объектов (например, поликлиник, школ и т. д.) для повышения обеспеченности городским сервисом населения рассматриваемой территории.

## 🚀 Установка

### 1. Библиотека может быть установлена с помощью pip:

```bash
pip install git+shttps://github.com/ValeriyaTikhevich/solver_FLP.git@main
```

### 2. Либо можно склонировать репозиторий на Ваш локальный компьютер:

```bash
git clone https://github.com/ваш-аккаунт/solver_FLP.git
cd solver_FLP
```

Создать виртуальное окружение:

#### Для Windows:
```bash
make venv
.venv\Scripts\activate
```

#### Для macOS/Linux::
```bash
make venv
source .venv/bin/activate
```

И установить зависимости:
```bash
pip install .
```

## 🧑‍💻 Использование

После установки Вы можете импортировать библиотеку и начать использовать её для решения задач:

```bash
from method import genetic_algorithm_main, choose_edges, fitness_plot
```

Примеры использования библиотеки и формата входных данных лежат в разделе **examples**

