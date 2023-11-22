default: build

help:
	@echo 'Management commands for fg_hgcl:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the fg_hgcl project.'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t fg_hgcl 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name fg_hgcl -v `pwd`:/workspace/fg_hgcl fg_hgcl:latest /bin/bash

up: build run

rm: 
	@docker rm fg_hgcl

stop:
	@docker stop fg_hgcl

reset: stop rm