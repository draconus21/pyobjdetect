# pyObjDetect


## Prerequisites
* [Cuda Toolkit 11.7](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)


## Environment setup
First, run the `source scripts/env.sh` to get setup. It will ask to setup a python virtual environment if one does not already exist
```bash
# to use the default virtual environment (.env)
source ./scripts/env.sh

# if you want to use your custom virtual environment
source ./scripts/env.sh <path_to_custom venv>
```

Then, install the package in editable mode
```bash
./scripts/build.sh install
```

That's it!

## Developping
When you return to the project, run the `source` command to set up the environment and reactivate your virtual environment
```bash
# to use the default virtual environment (.env)
source ./scripts/env.sh

# if you want to use your custom virtual environment
source ./scripts/env.sh <path_to_custom venv>
```
