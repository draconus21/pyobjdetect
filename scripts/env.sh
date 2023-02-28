#! /bin/bash


# -o prevent errors from being masked
# -u require vars to be declared before referencing them
set -o pipefail

normal=$(tput sgr0)
bg_normal=$(tput setab sgr0)
bg_black=$(tput setab 0)
red=$(tput setaf 1)
green=$(tput setaf 2)
yellow=$(tput setaf 3)
cyan=$(tput setaf 6)

function green () {
    echo -e ${green}$1${normal}
}

function red () {
    echo -e ${red}$1${normal}
}

function yellow () {
    echo -e ${yellow}$1${normal}
}

function cyan () {
  echo -e ${cyan}$1${normal}
}

DEFAULT_LD_LIBRARY_PATH=""
LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-$DEFAULT_LD_LIBRARY_PATH}

# ------------------------------------------------------------------- #
#                                 MUNGERS                             #
# ------------------------------------------------------------------- #

# if directory doesn't exist in a PATH environmental variable, create it.

path_munge () {
  if ! echo "$PATH" | /bin/grep -Eq "(^|:)$1($|:)" ; then
	yellow "$1 ${normal} doesn't exist. ${cyan}Adding to PATH."
    PATH="$1:$PATH"
  fi
}

ld_library_path_munge () {
  if ! echo "$LD_LIBRARY_PATH" | /bin/grep -Eq "(^|:)$1($|:)" ; then
	yellow "$1 ${normal} doesn't exist. ${cyan}Adding to LD_LIBRARY_PATH."
    LD_LIBRARY_PATH="$1:$LD_LIBRARY_PATH"
  fi
}

# ------------------------------------------------------------------- #
#                            HELPER FUNCTIONS                         #
# ------------------------------------------------------------------- #

# Adds visual padding for visibility
# usage - output_spacing "command" "info"
tmp_padding="                                    " # expand as necessary...
function output_spacing () {
	tmp_stringToPad=$1
	printf "%s%s %s %s\n" "${yellow}$1" "${tmp_padding:${#tmp_stringToPad}}" ":" "${normal}$2"
}

# outputs received script arguments
# SYNTAX - output_received_args "$@"
function output_received_args () {

  echo -n "$bg_black""$yellow"  # set bg/text colors
  printf "[cmd] : %s" "${0##*/}"
  args=("$@")           # store arguments in a special array
  ELEMENTS=${#args[@]}  # get number of elements

  for (( v=0;v<$ELEMENTS;v++)); do
    printf ' %s' "${args[${v}]}"
  done

  echo -n "$bg_normal""$normal" # reset bg/text colors
  printf "\n\n"
}

# exports an array of env. vars
# usage - export_env_var_arrays "array"
function export_env_var_arrays() {
	arr=("$@")
	for i in "${arr[@]}"; do
		export "${i?}"
	done
}

# displays an array of env. vars
# usage - display_env_var_arrays "array"
function display_env_var_arrays() {
	arr=("$@")

	for i in "${arr[@]}"; do
		case $SHELL in
		*/zsh) # shell-check doesn't support zsh and will mark as error
		output_spacing "${i}=" "${(P)i}"
		;;
		*/bash) # ${!i} is incompatible on zsh (indirect expansion)
		output_spacing "${i}=" "${!i}"
		;;
		*)
		echo "no compatible shells"
		esac

	done
}

make_dir() {
    if [[ ! -d $1 ]]; then
    echo "creating dir: $1"
        mkdir -p $1
    fi
}

green "═══ depth-metric env.sh ═══"
output_received_args "$@"

# ------------------------------------------------------------------- #
#                                  ARGS                               #
# ------------------------------------------------------------------- #
NXK_REPO_DIR=""

# ------------------------------------------------------------------- #
#                             BASE Env.Vars                           #
# ------------------------------------------------------------------- #
green "\n--- Base Env.Vars ---"
if [[ -z "$NXK_REPO_DIR" ]]; then
        NXK_REPO_DIR=$(pwd)
        echo "NXK_REPO_DIR is empty: assuming NXK_REPO_DIR: ($NXK_REPO_DIR)";

fi

if [[ ! -d "$NXK_REPO_DIR" ]]; then
        red "Error: NXK_REPO_DIR:$NXK_REPO_DIR doesn't point to a valid directory";
        return 1;
fi

chmod u+x -R "${NXK_REPO_DIR}/scripts"

NXK_DATA_DIR="$NXK_REPO_DIR/data"
NXK_EXPERIMENTS_DIR="$NXK_REPO_DIR/experiments"
NXK_LOG_DIR="$NXK_REPO_DIR/logs"
NXK_LOG_CFG="$NXK_REPO_DIR/default-logging.json"

make_dir $NSK_DATA_DIR
make_dir $NXK_EXPERIMENTS_DIR
make_dir $NXK_LOG_DIR

cyan "\n[Generated Base Env.Vars]"
arrayEnvVarsToExport=(  NXK_REPO_DIR
                        NXK_DATA_DIR
                        NXK_EXPERIMENTS_DIR
                        NXK_LOG_DIR
                        NXK_LOG_CFG)

export_env_var_arrays "${arrayEnvVarsToExport[@]}"
display_env_var_arrays "${arrayEnvVarsToExport[@]}"

# ------------------------------------------------------------------- #
#                                 PYTHON                              #
# ------------------------------------------------------------------- #
green "\n--- Python Env.Vars ---"


if [[ "$OSTYPE" == "msys" ]]; then
  NKK_PYTHON_EXECUTABLE=$(which python)
else
  NXK_PYTHON_EXECUTABLE=$(which python3)
fi

DEFAULT_VENV=".env"

# use default venv if first arg is empty
arg1=${1:-""}
VAR=$1
NXK_PYTHON_VENV="${VAR:=${NXK_REPO_DIR}/${DEFAULT_VENV}}"

if [[ "$OSTYPE" == "msys" ]]; then
  NXK_PYTHON_VENV_PATH="${NXK_PYTHON_VENV}"/Scripts/activate
else
  NXK_PYTHON_VENV_PATH="${NXK_PYTHON_VENV}"/bin/activate
fi

# check if venv dir exists, if not create one after confirming with user
if [[ ! -d ${NXK_PYTHON_VENV} ]]; then
  red "virtual env does not exist at ${NXK_PYTHON_VENV}"
  vared -p "Would you like me to create one? [y/n]: " -c confirm
  if [[ "$confirm" == "y" ]]; then
    yellow "creating venv ${NXK_PYTHON_VENV}"
    "${NXK_PYTHON_EXECUTABLE}" -m venv "${NXK_PYTHON_VENV}"
  fi
fi

source ${NXK_PYTHON_VENV_PATH}

# get python executable from venv
NXK_PYTHON_EXECUTABLE=$(which python)
NXK_PYTHON_VERSION=$($NXK_PYTHON_EXECUTABLE -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

arrayEnvVarsToExport=(  NXK_PYTHON_VENV
                        NXK_PYTHON_VENV_PATH
                        NXK_PYTHON_EXECUTABLE
                        NXK_PYTHON_VERSION)

export_env_var_arrays "${arrayEnvVarsToExport[@]}"
display_env_var_arrays "${arrayEnvVarsToExport[@]}"

echo "${cyan}Active Python Exec in use : ${normal} ${NXK_PYTHON_EXECUTABLE} version(${NXK_PYTHON_VERSION})"


echo "env    : ${NXK_PYTHON_VENV_PATH}"
echo "python : ${NXK_PYTHON_EXECUTABLE}"
echo "version: ${NXK_PYTHON_VERSION}"


green "\n--- Final Env.Vars ---"
cyan "(All env. vars. generated from this script related to NXK)"
$NXK_REPO_DIR/scripts/log-env-variables.sh

green "═══ end env.sh ═══"

# To avoid propagating the unbound and pipefail to the current terminal.
set +uo pipefail
