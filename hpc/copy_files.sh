#!/bin/bash

rsync -av --exclude-from=.copy-ignore $1 "ejh19@login.hpc.ic.ac.uk:~/$2"
