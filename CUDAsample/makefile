# Makefile to delete all files except .cu files, config_turing_islip.icnt, gpgpusim.config, and the Makefile itself
# And to copy specific configuration files for simulation

# Target directory
DIR = /home/lzydaphne/CUDAsample

# Source directory for GPGPU-Sim configuration files
GPGPU_SIM_DIR = ../gpgpu-sim_distribution/configs/tested-cfgs/SM75_RTX2060

# Files to keep explicitly
KEEP_FILES = config_turing_islip.icnt gpgpusim.config makefile

.PHONY: clean copy

# Target to clean the directory
clean:
	@echo "Cleaning up..."
	@rm -f _cuobjdump_list_ptx_*
	@rm -f _app_cuda_version_*

# Target to copy configuration files
copy:
	@cp $(GPGPU_SIM_DIR)/config_turing_islip.icnt $(DIR)/
	@cp $(GPGPU_SIM_DIR)/gpgpusim.config $(DIR)/
