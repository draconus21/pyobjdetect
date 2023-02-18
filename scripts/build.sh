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

green "==== build.sh ===="

function install() {
    green "install pyobjdetect"

    # install package locally
    pip install --upgrade pip
    pip install -e $NXK_REPO_DIR[dev]
}

case "${1}" in
    -h|--help)
        help="  -h: Print this message\n"
        help+="  install: install depth metric module\n"
        printf "%b" "usage: ./build.sh <options> \n$help\n"
        exit 0
        ;;
    install|*)
        install
        exit 0
        ;;
esac

green "==== build.sh ===="
