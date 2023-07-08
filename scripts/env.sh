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
  printf "[cmd] : %s" "${SHELL##*/}"
  args=("$@")           # store arguments in a special array
  ELEMENTS=${#args[@]}  # get number of elements

  case $SHELL in
    */zsh)
    START=1
    END=$ELEMENTS+1
    ;;
    *)
    START=0
    END=$ELEMENTS
    ;;
  esac

  for (( v=$START;v<$END;v++)); do
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
		output_spacing "${i}" "${(P)i}"
		;;
		*/bash) # ${!i} is incompatible on zsh (indirect expansion)
		output_spacing "${i}" "${!i}"
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

green "═══ start pyobjdetect env.sh ═══"
output_received_args "$@"

# ------------------------------------------------------------------- #
#                                  ARGS                               #
# ------------------------------------------------------------------- #
ODT_DIR="$(pwd)"

# ------------------------------------------------------------------- #
#                             BASE Env.Vars                           #
# ------------------------------------------------------------------- #

if [[ ! -d "$ODT_DIR" ]]; then
        red "Error: ODT_DIR:$ODT_DIR doesn't point to a valid directory";
        return 1;
fi

chmod u+x -R "${ODT_DIR}/scripts"

ODT_DATA_DIR="$ODT_DIR/data"
ODT_EXPERIMENTS_DIR="$ODT_DIR/experiments"
ODT_LOG_DIR="$ODT_DIR/logs"
ODT_LOG_CFG="$ODT_DIR/default-logging.json"

make_dir $ODT_DATA_DIR
make_dir $ODT_EXPERIMENTS_DIR
make_dir $ODT_LOG_DIR

cyan "\n[Generated Base Env.Vars]"
arrayEnvVarsToExport=(  ODT_DIR
                        ODT_DATA_DIR
                        ODT_EXPERIMENTS_DIR
                        ODT_LOG_DIR
                        ODT_LOG_CFG)

export_env_var_arrays "${arrayEnvVarsToExport[@]}"
display_env_var_arrays "${arrayEnvVarsToExport[@]}"

# ------------------------------------------------------------------- #
#                                 PYTHON                              #
# ------------------------------------------------------------------- #
green "\n--- Python Env.Vars ---"


if [[ "$OSTYPE" == "msys" ]]; then
  NKK_PYTHON_EXECUTABLE=$(which python)
else
  ODT_PYTHON_EXECUTABLE=$(which python3)
fi

DEFAULT_VENV=".env"

# use default venv if first arg is empty
arg1=${1:-""}
VAR=$1
ODT_PYTHON_VENV="${VAR:=${ODT_DIR}/${DEFAULT_VENV}}"

if [[ "$OSTYPE" == "msys" ]]; then
  ODT_PYTHON_VENV_PATH="${ODT_PYTHON_VENV}"/Scripts/activate
else
  ODT_PYTHON_VENV_PATH="${ODT_PYTHON_VENV}"/bin/activate
fi

# check if venv dir exists, if not create one after confirming with user
if [[ ! -d ${ODT_PYTHON_VENV} ]]; then
    red "virtual env does not exist at ${ODT_PYTHON_VENV}"
    case $SHELL in
    */zsh)
        vared -p "Would you like me to create one? [y/n]: " -c confirm
    ;;
    */bash) # vared incompatible on bash
        echo "Would you like me to create one? [y/n]: "; read confirm
    ;;
    *)
    echo "no compatible shells"
    esac
  if [[ "$confirm" == "y" ]]; then
    yellow "creating venv ${ODT_PYTHON_VENV}"
    "${ODT_PYTHON_EXECUTABLE}" -m venv "${ODT_PYTHON_VENV}"
  fi
fi

source ${ODT_PYTHON_VENV_PATH}

# get python executable from venv
ODT_PYTHON_EXECUTABLE=$(which python)
ODT_PYTHON_VERSION=$($ODT_PYTHON_EXECUTABLE -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')

arrayEnvVarsToExport=(  ODT_PYTHON_VENV
                        ODT_PYTHON_VENV_PATH
                        ODT_PYTHON_EXECUTABLE
                        ODT_PYTHON_VERSION)

export_env_var_arrays "${arrayEnvVarsToExport[@]}"
display_env_var_arrays "${arrayEnvVarsToExport[@]}"

green "\n--- Final Env.Vars ---"
cyan "(All env. vars. generated from this script related to ODT)"
$ODT_DIR/scripts/log-env-variables.sh

green "═══ end pyobjdetect env.sh ═══"

# To avoid propagating the unbound and pipefail to the current terminal.
set +uo pipefail