# Makefile for Docker Compose commands

# Start Docker Compose services in detached mode
up:
	docker compose up -d

# Remove all Docker images
rmi:
	docker rmi -f $(docker images -aq)

# Stop and remove Docker Compose services
down:
	docker compose down

# Run all commands in reverse order: down, rmi, up
all:
	$(MAKE) down
	$(MAKE) rmi
	$(MAKE) up

# Phony targets to avoid conflicts with file names
.PHONY: up rmi down all
