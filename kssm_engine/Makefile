CLANG ?= clang
CC    ?= gcc

all: bpf/kssm_engine.bpf.o bin/kssm_daemon

bpf/kssm_engine.bpf.o: bpf/kssm_engine.bpf.c include/kssm_abi.h
	$(CLANG) -O2 -g -target bpf -D__TARGET_ARCH_x86 -c $< -o $@

bin/kssm_daemon: src/kssm_daemon.c include/kssm_abi.h
	@mkdir -p bin
	$(CC) -O2 $< -o $@ -lbpf -lelf -lz -lm

verify: all
	sudo ./bin/kssm_daemon --verify

clean:
	rm -rf bpf/kssm_engine.bpf.o bin/kssm_daemon
