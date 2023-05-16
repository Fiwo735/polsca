user=$(if $(shell id -u),$(shell id -u),9001)
group=$(if $(shell id -g),$(shell id -g),1000)
phism=/workspace
vhls=/scratch/shared/Xilinx/
th=1
example=2mm

# Steps to start on Beholder server:
# $ vagrant up && vagrant ssh
# $ cd /vagrant
# $ make shell

# To run:
# $ source scripts/source-all.sh && make ... # (e.g. test-emit)

# To build:
# $ source ~/.bashrc && ./scripts/build-phism.sh

# Searching LLVM:
# grep ./polygeist/llvm-project/mlir/lib/ ./polygeist/llvm-project/mlir/include/mlir/ -nRe "pattern"

.PHONY: build-docker shell install-pyphism test-example test-emit test-polybench test-polybench-polymer build-phism sync clean clean-phism

# Build docker container
build-docker: 
	(cd Docker; docker build --build-arg UID=$(user) --build-arg GID=$(group) --build-arg VHLS_PATH=$(vhls) . --tag phism8)

# Enter docker container
shell: build-docker
	docker run -it -v $(shell pwd):/workspace -v $(vhls):$(vhls) phism8:latest /bin/bash
	
test-example:
	python3 scripts/pb-flow.py ./example/polybench -e $(example) --work-dir ./tmp/phism/pb-flow.tmp --cosim

test-emit:
	python3 scripts/pb-flow.py ./example/polybench -e 2mm --work-dir ./tmp/phism/pb-flow.tmp --skip-vitis --emit-hls --loop-transforms --array-partition

# Evaluate polybench (baseline) - need to be used in environment
test-polybench:
	python3 scripts/pb-flow.py -c -j $(th) example/polybench

# Evaluate polybench (polymer) - need to be used in environment
test-polybench-polymer:
	python3 scripts/pb-flow.py -c -p -j $(th) example/polybench

# Build LLVM and Phism - temporary fix for missing mlir-clang
build-phism:
	./scripts/build-llvm.sh
	./scripts/build-polygeist.sh
	(cd ./polygeist/build; make mlir-clang)
	./scripts/build-polymer.sh
	./scripts/build-phism.sh

install-pyphism:
	pip install --upgrade ./pyphism/

# Sync and update submodules
sync:
	git submodule sync
	git submodule update --init --recursive

clean: clean-phism
	rm -rf $(phism)/llvm/build

clean-phism:
	rm -rf $(phism)/build