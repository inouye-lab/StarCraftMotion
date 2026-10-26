# Makefile for the StarCraft Motion processing pipeline.
#
#   make install   -- bootstrap conda env + SC2 binary + maps + replay pack 1
#   make environment   -- create the conda env only
#   make maps replays  -- download Blizzard map packs / replay pack 1 only
#   make clean-all     -- remove SC2, maps, replays, conda env
#
# Override defaults on the command line, e.g.
#   make install INSTALL_DIR=/opt/sc2-data CONDA_INSTALL_DIR=/opt/conda

SHELL := /usr/bin/env bash

# Where SC2, maps and replay packs are installed. Defaults to ./datasets in the
# current working directory; nothing is hardcoded to a specific machine.
INSTALL_DIR ?= $(CURDIR)/datasets

# Where Miniconda3 lives. Defaults to $HOME/miniconda3 (portable across users).
CONDA_INSTALL_DIR ?= $(HOME)

CONDA := $(CONDA_INSTALL_DIR)/miniconda3/bin/conda
ENV_NAME := starcraft

.PHONY: dummy install environment maps replays clean-all clean-environment \
        clean-miniconda3 clean-StarCraftII clean-maps clean-replays \
        export-environment

dummy:
	@echo "Use 'make install' to install everything,"
	@echo "or 'make environment' / 'make maps' / 'make replays' to install just one piece."
	@echo "Override paths with INSTALL_DIR=... and CONDA_INSTALL_DIR=..."

install: environment maps replays | $(INSTALL_DIR)/StarCraftII
	@echo
	@echo "=== Install complete ==="
	@echo "Activate the env: conda activate $(ENV_NAME)"
	@echo "Then export:      export SC2PATH=$(INSTALL_DIR)/StarCraftII"

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------

environment: | $(CONDA_INSTALL_DIR)/miniconda3/envs/$(ENV_NAME)
	@echo "Conda env '$(ENV_NAME)' is ready."

$(CONDA_INSTALL_DIR)/miniconda3:
	wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	bash Miniconda3-latest-Linux-x86_64.sh -b -p $(CONDA_INSTALL_DIR)/miniconda3
	rm Miniconda3-latest-Linux-x86_64.sh

$(CONDA_INSTALL_DIR)/miniconda3/envs/$(ENV_NAME): environment.yml | $(CONDA_INSTALL_DIR)/miniconda3
	$(CONDA) env create -f environment.yml

# ---------------------------------------------------------------------------
# StarCraft II binary, maps, replays
#
# All artifacts come from the same Blizzard CDN URLs referenced by
# https://github.com/Blizzard/s2client-proto. The EULA password
# 'iagreetotheeula' is documented by Blizzard for these archives.
# ---------------------------------------------------------------------------

$(INSTALL_DIR):
	mkdir -p $(INSTALL_DIR)

$(INSTALL_DIR)/StarCraftII: | $(INSTALL_DIR)
	wget http://blzdistsc2-a.akamaihd.net/Linux/SC2.3.16.1.zip
	unzip -P iagreetotheeula SC2.3.16.1.zip -d $(INSTALL_DIR)
	rm SC2.3.16.1.zip

maps: | $(INSTALL_DIR)/StarCraftII/Maps/Ladder2017Season1 \
       $(INSTALL_DIR)/StarCraftII/Maps/Melee

$(INSTALL_DIR)/StarCraftII/Maps/%: | $(INSTALL_DIR)/StarCraftII
	wget http://blzdistsc2-a.akamaihd.net/MapPacks/$(@F).zip \
	     -O $(INSTALL_DIR)/StarCraftII/Maps/$(@F).zip
	unzip -P iagreetotheeula $(INSTALL_DIR)/StarCraftII/Maps/$(@F).zip \
	      -d $(INSTALL_DIR)/StarCraftII/Maps/
	rm $(INSTALL_DIR)/StarCraftII/Maps/$(@F).zip

replays: | $(INSTALL_DIR)/StarCraftII/Replays-3.16.1-Pack_1-fix

$(INSTALL_DIR)/StarCraftII/Replays-%: | $(INSTALL_DIR)/StarCraftII
	wget http://blzdistsc2-a.akamaihd.net/ReplayPacks/$*.zip \
	     -O $(INSTALL_DIR)/StarCraftII/$*.zip
	unzip -P iagreetotheeula $(INSTALL_DIR)/StarCraftII/$*.zip \
	      -d $(INSTALL_DIR)/StarCraftII/
	mv $(INSTALL_DIR)/StarCraftII/Replays $(INSTALL_DIR)/StarCraftII/Replays-$*
	rm $(INSTALL_DIR)/StarCraftII/$*.zip

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean-all: clean-environment clean-StarCraftII clean-maps clean-replays

clean-environment:
	-$(CONDA) env remove --name $(ENV_NAME) --yes

clean-miniconda3:
	-rm -rf $(CONDA_INSTALL_DIR)/miniconda3

clean-StarCraftII:
	-rm -rf $(INSTALL_DIR)/StarCraftII

clean-maps:
	-rm -rf $(INSTALL_DIR)/StarCraftII/Maps/*

clean-replays:
	-rm -rf $(INSTALL_DIR)/StarCraftII/Replays*

export-environment:
	$(CONDA) env export --from-history --name $(ENV_NAME)
