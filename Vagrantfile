# -*- mode: ruby -*-
# vi: set ft=ruby :
# All Vagrant configuration is done below. The "2" in Vagrant.configure
# configures the configuration version (we support older styles for
# backwards compatibility). Please don't change it unless you know what
# you're doing.
Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/focal64"
  config.vm.synced_folder "..", "/vagrant-workspace"
  config.vm.synced_folder "/scratch/shared/Xilinx/", "/scratch/shared/Xilinx/"
  # Require plugin https://github.com/leighmcculloch/vagrant-docker-compose
  config.vagrant.plugins = "vagrant-docker-compose"
  # Install docker and docker-compose
  config.vm.provision :docker
  config.vm.provision :docker_compose
  config.vm.provider "virtualbox" do |vb|
      # Customize the amount of memory on the VM:
      vb.cpus = 16
      vb.memory = "32768"
  end
end