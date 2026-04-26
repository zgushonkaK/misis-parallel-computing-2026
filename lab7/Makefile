# =====================================================================
# Makefile для лабораторной работы №7, вариант 8.
# Цель — Apple Silicon (ARM64, NEON).
#
# Основные команды:
#   make            — собрать программу
#   make run        — собрать и запустить на test.bmp
#   make test-bmp   — сгенерировать тестовый BMP (test.bmp)
#   make clean      — удалить бинарник и выходные файлы
#
# Требования: Xcode Command Line Tools (ставятся командой xcode-select --install).
# NEON на ARM64 встроен в архитектуру — никаких -m... флагов не требуется.
# =====================================================================

CXX      := clang++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra

TARGET   := blur4x4
SOURCES  := main.cpp
TEST_BMP := test.bmp

.PHONY: all run test-bmp clean

all: $(TARGET)

$(TARGET): $(SOURCES)
	@echo "=== Сборка для Apple Silicon (ARM64 / NEON) ==="
	@echo "CXX      = $(CXX)"
	@echo "CXXFLAGS = $(CXXFLAGS)"
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $(TARGET)
	@echo ""
	@echo "Готово: ./$(TARGET)"

# Сгенерировать тестовый 24-bit BMP (требуется python3 — есть в macOS по умолчанию).
test-bmp: $(TEST_BMP)

$(TEST_BMP): make_test_bmp.py
	python3 make_test_bmp.py

# Собрать и запустить на test.bmp (создаст test.bmp при необходимости).
run: $(TARGET) $(TEST_BMP)
	./$(TARGET) $(TEST_BMP) --repeats 30

clean:
	rm -f $(TARGET) \
	      *_gray.bmp *_blur_scalar.bmp *_blur_neon.bmp
	@echo "Очищено."
