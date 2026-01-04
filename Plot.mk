# Copyright (c) 2023 Gilad Odinak

CFLAGS += -DHAS_PLOT

MPL_CC = $(CPPC)
MPL_CFLAGS = $(CFLAGS) -std=c++11 -Wno-deprecated-declarations -fpermissive 
MPL_CFLAGS += -DWITHOUT_NUMPY -Wno-unused-parameter

PLOT_DIR = $(SRC_DIR)/plot

PLOT_INC = -I$(shell python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')
ifeq ($(OSTYPE),macos)
PLOT_LIBS = -L$(shell python3 -c 'import sysconfig; print(sysconfig.get_config_var("LIBPL"))')
else
PLOT_LIBS = -L$(shell python3 -c 'import sysconfig; print(sysconfig.get_path("stdlib"))')
endif
PY_LIB = -lpython$(shell python3 -c 'import sysconfig; print(sysconfig.get_python_version())')

INC_DIRS += $(PLOT_INC)
LIB_DIRS += $(PLOT_LIBS)
LIBS += -lstdc++ $(PY_LIB)
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(MPL_CC) $(MPL_CFLAGS) $(INC_DIRS) -c -o $@ $<

OBJS += $(BUILD_DIR)/plot/plot.o
