#! /bin/bash

normal=$(tput sgr0)
red=$(tput setaf 1)
yellow=$(tput setaf 3)
green=$(tput setaf 2)

function green () {
    echo ${green}$1${normal}
}

function yellow() {
    echo ${yellow}$1${normal}
}

function red () {
    echo ${red}$1${normal}
}

if [[ -n "${NXK_PYTHON_VENV}" ]]; then
    if [[ -d "${NXK_PYTHON_VENV}" ]]; then
        green "removing ${NXK_PYTHON_VENV}"
        rm -r "${NXK_PYTHON_VENV}"/
    fi
fi

yellow "make sure to run source ./scripts/env.sh from a new terminal to setup the environment properly"