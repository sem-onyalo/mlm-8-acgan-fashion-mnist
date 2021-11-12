#!/bin/bash

set -e # exit when any command fails

wrkdir="."
venvcmd="env/Scripts/activate"
runcmd="python main.py --summary"

log()
{
    echo [$(date)] "$1"
}

while getopts "w:v:r:" args; do
  case $args in
    w) wrkdir=$OPTARG;;
    v) venvcmd=$OPTARG;;
    r) runcmd=$OPTARG;;
  esac
done

log "Switching to working directory '$wrkdir'"
cd $wrkdir

log "Activating virtual environment '$venvcmd'"
# source $venvcmd

python -m venv env

source env/bin/activate

pip install -r requirements.txt

log "Running '$runcmd'"
$runcmd
