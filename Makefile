CXX=g++
CXXFLAGS=-I. -std=c++11 -O3
TARGET=test
SRC=baseline.cpp


all: $(TARGET)

$(TARGET): baseline.cpp 
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET)