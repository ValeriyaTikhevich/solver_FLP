.PHONY: install clean build venv clean-build

# Установка зависимостей
install:
	@echo "Installing dependencies from pyproject.toml..."
	pip install .

venv: #then .venv\Scripts\activate
	python3 -m venv .venv

# Очистка временных файлов
clean:
	@echo "Cleaning up..."
	@if exist __pycache__ rmdir /S /Q __pycache__
	@if exist build rmdir /S /Q build
	@if exist dist rmdir /S /Q dist
	@if exist *.egg-info for /D %%i in (*.egg-info) do rmdir /S /Q "%%i"

# Сборка пакета
build:
	@echo "Building the Python package..."
	python -m build

# Удаление сборок
clean-build:
	@echo "Cleaning build artifacts..."
	@if exist build rmdir /S /Q build
	@if exist dist rmdir /S /Q dist
	@if exist *.egg-info for /D %%i in (*.egg-info) do rmdir /S /Q "%%i"