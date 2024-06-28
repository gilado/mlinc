# Copyright (c) 2023 Gilad Odinak

CFLAGS += -DHAS_PLOT

MPL_CC = g++
MPL_CFLAGS = $(CFLAGS) -std=c++11 -Wno-deprecated-declarations -fpermissive

PLOT_DIR = $(SRC_DIR)/plot

ifeq ($(OSTYPE),macos)
MPL_CFLAGS += -DWITHOUT_NUMPY
OSPFX = /usr/local/Cellar
PYPFX = $(OSPFX)/python@3.11/3.11.7_1/Frameworks/Python.framework/Versions/3.11
PLOT_INC =  -I$(PYPFX)/include/python3.11
PLOT_LIBS = -L$(PYPFX)/lib/python3.11/config-3.11-darwin/
PY_LIB = -lpython3.11
else
MPL_CFLAGS += -DWITHOUT_NUMPY
PLOT_INC = -I/usr/include//python3.10/
PLOT_LIBS = -L/usr/lib64/python3.10/
PY_LIB = -lpython3.10
endif

INC_DIRS += $(PLOT_INC)
LIB_DIRS += $(PLOT_LIBS)
LIBS += -lstdc++ $(PY_LIB)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(MPL_CC) $(MPL_CFLAGS) $(PLOT_INC) -c -o $@ $<

OBJS += $(BUILD_DIR)/plot/plot.o
