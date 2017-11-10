#!/bin/bash

function usage()
{
    echo "Usage: $0 <username> <password> [<file path>] [<comment>]"
}

[ -z "$1" -o -z "$2" ] && usage && exit
kg config -c kktv-data-game-1711 -u $1 -p $2

echo "Submissions:"
kg submissions

[ -z "$3" -o -z "$4" ] && exit
kg submit $3 -c kktv-data-game-1711 -u $1 -p $2 -m "$4"
