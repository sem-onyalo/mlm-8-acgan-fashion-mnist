#!/bin/bash

set -e # exit when any command fails

wrkdir="."
venvcmd="env/bin/activate"
runcmd="python3 main.py --summary"

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
source $venvcmd

log "Running '$runcmd'"
$runcmd
