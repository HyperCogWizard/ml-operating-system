# Makefile for Robotics Middleware

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -D_GNU_SOURCE
LDFLAGS = -lpthread -lm

SRCDIR = .
OBJDIR = obj
BINDIR = .

SOURCES = main.c robotics_middleware.c tensor_abstraction.c gguf_integration.c hypergraph.c config.c
OBJECTS = $(SOURCES:%.c=$(OBJDIR)/%.o)
TARGET = robotics-middleware

.PHONY: all clean install

all: $(BINDIR)/$(TARGET)

$(BINDIR)/$(TARGET): $(OBJECTS) | $(BINDIR)
	$(CC) $(OBJECTS) -o $@ $(LDFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c | $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BINDIR):
	mkdir -p $(BINDIR)

clean:
	rm -rf $(OBJDIR)
	rm -f $(BINDIR)/$(TARGET)

install: $(BINDIR)/$(TARGET)
	install -D -m 0755 $(BINDIR)/$(TARGET) $(DESTDIR)/usr/bin/$(TARGET)

# Dependencies
$(OBJDIR)/main.o: main.c robotics_middleware.h tensor_abstraction.h
$(OBJDIR)/robotics_middleware.o: robotics_middleware.c robotics_middleware.h tensor_abstraction.h gguf_integration.h
$(OBJDIR)/tensor_abstraction.o: tensor_abstraction.c tensor_abstraction.h robotics_middleware.h
$(OBJDIR)/gguf_integration.o: gguf_integration.c gguf_integration.h robotics_middleware.h tensor_abstraction.h
$(OBJDIR)/hypergraph.o: hypergraph.c robotics_middleware.h
$(OBJDIR)/config.o: config.c robotics_middleware.h