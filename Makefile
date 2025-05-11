# Переменные
VENV_DIR := .venv
PYTHON := python3
UV := $(VENV_DIR)/bin/uv
PYTHON_BIN := $(VENV_DIR)/bin/python

# Основная цель
.PHONY: all
all: $(UV)
	@echo "✅ Установка зависимостей с помощью uv..."
	$(UV) sync
	$(UV) pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# Цель: создать виртуальное окружение и установить uv
$(UV): | $(VENV_DIR)
	@echo "📦 Установка uv в виртуальное окружение..."
	$(VENV_DIR)/bin/pip install uv

# Цель: создать виртуальное окружение
$(VENV_DIR):
	@echo "🐍 Создание виртуального окружения..."
	$(PYTHON) -m venv $(VENV_DIR)