#!/bin/bash

CMD="nosetests syris/tests"
NO_PMASF=$CMD" -a '!pmasf_required'"

if [ $# -eq 0 ];
then
  $CMD
else
  if [ $1 = "np" ];
    then
      eval $NO_PMASF
    else
      echo "Unrecognized input."
  fi
fi

