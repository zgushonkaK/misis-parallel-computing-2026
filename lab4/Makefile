CXX = g++
CXXFLAGS = -std=c++14 -O2 -Wall -Wextra

TARGET = cache_latency
SOURCES = main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
BUILD_DIR = build

.PHONY: all clean

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(BUILD_DIR)/$(OBJECTS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(BUILD_DIR)/$(OBJECTS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)
