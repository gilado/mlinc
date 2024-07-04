# Copyright (c) 2023 Gilad Odinak

ifeq ($(shell uname), Linux)
OSTYPE=linux
else ifeq ($(shell uname), Darwin)
OSTYPE=macos
else
$(error Unsupported OS)
endif

ifeq ($(DEBUG),)    # DEBUG is blank - not debug 
ifeq ($(MARCH),)
MARCH = native
endif
CFLAGS += -O3 -march=$(MARCH)
else
CFLAGS += -ggdb     # gdb support
endif

ifneq ($(MEMCHK),)  # MEMCHK is not blank - add memory error detector
CFLAGS += -fsanitize=address
LFLAGS += -static-libasan
endif

ifneq ($(PROFILE),) # PROFILE is not blank - add profilng support
CFLAGS += -pg -fno-inline
LFLAGS += -pg
endif

ifneq ($(USEDOUBLE),) # USEDOUBLE is not blank - use double instead of float
    CFLAGS += -DUSE_DOUBLE # bit exact math with python
endif

ifeq ($(OSTYPE),macos)
LFLAGS += -w # Suppress OS version message spam on macos
endif

