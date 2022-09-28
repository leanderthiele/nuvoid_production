#!/bin/bash

# A small collection of useful utility functions

# Print to stderr
# Arguments:
#   message: the string to print
function utils::printerr {
  printf "%s\n" "$*" >&2
  return 0
}

# Replace a tag in a configuration file
# Arguments:
#   file: the file to operate on (in place)
#   tag: the tag to be replaced (expected to be found in [lead]...[trail])
#   replacement: the replacement text
#   lead: [optional] leading indicactor. Default '<<<'
#   trail: [optional] trailing indicator. Default '>>>'
# Returns:
#   None
function utils::replace {
  file="$1"
  tag="$2"
  replacement="$3"
  lead="$4" # may be empty
  trail="$5" # may be empty

  if [ -z "$lead" ]; then
    lead='<<<'
  fi

  if [ -z "$trail" ]; then
    trail='>>>'
  fi

  sed -i "s@${lead}${tag}${trail}@$replacement@g" $file

  return 0
}

# Hash a string (hexadecimal)
# Arguments:
#   string: the string to be hashed
# Returns:
#   hexadecimal hash string
function utils::hex_hash {
  string="$1"
  echo "$(echo "$string" | md5sum | cut -f1 -d" ")"
  return 0
}

# Hash a string (positive decimal)
# Arguments:
#   string: the string to be hashed
#   bits: [optional] restrict to this range
# Returns:
#   positive decimal hash string
function utils::dec_hash {
  string="$1"
  bits="$2" # may be empty
  hex_hash="$(utils::hex_hash "$string")"
  dec_hash="$(echo "ibase=16; ${hex_hash^^}" | bc)"
  dec_hash="${dec_hash#-}" # remove minus sign if present
  if [ "$bits" ]; then
    # do not use intrinsic shell arithmetic here since it is limited in value
    dec_hash=$(echo "$dec_hash % 2^$bits" | bc)
  fi
  echo "$dec_hash"
  return 0
}

# Convert a floating point expression to bc-readable notation with sufficient precision
# Arguments:
#   expression: the string to be converted
#   precision: [optional] number of after-comma digits used in %.Nf. Default is 16.
# Returns:
#   normalized string
function utils::normalize_expr {
  expression="$1"
  precision="$2"
  if [ -z "$precision" ]; then
    precision=16
  fi

  # we match floating point numbers in scientific and decimal notation
  # it's a bit of a pain to deal with things like 1. and .1 while not capturing everything
  # We do not match pure integers since these are often used as exponents and bc doesn't like floats
  # in exponents
  p_float_1='[[:digit:]]+\.[[:digit:]]*' # 0., 0.1
  p_float_2='\.[[:digit:]]+' # .1,
  p_exp='[e,E][\+,\-]?[[:digit:]]+' # e1, E1, e+02, E-1 etc
  p_int='[[:digit:]]+'

  p_float="(${p_float_1})|(${p_float_2})"
  pattern="((${p_float})($p_exp)?)|(${p_int}${p_exp})"

  # these cannot be used in the passed expression, should be safe
  use_as_IFS='|'
  my_printf='printf'

  # make sure no shell expansion on * etc
  set -o noglob

  # this is a bit hacky but I don't know any other way
  # make sure * etc don't give us problems
  cmds="$(echo "$expression" \
          | sed -E 's@('"$pattern"')@'"$use_as_IFS""$my_printf"' "%.'"$precision"'f" \1'"$use_as_IFS"'@g')"

  out=""
  IFS="$use_as_IFS"
  for cmd in $cmds; do
    if [[ "${cmd:0:${#my_printf}}" == "$my_printf" ]]; then
      out="${out}$(eval "$cmd")"
    else
      out="${out}${cmd}"
    fi
  done

  # clean up!
  IFS=' '
  set +o noglob

  echo "$out"
  return 0
}

# Evaluate a floating point expression
# Arguments:
#   expression: a string containing a floating point expression to be evaluated
#   format: [optional] return value in this format. Default is .8f.
# Returns:
#   result string
function utils::feval {
  expression="$1"
  format="$2" # may be empty
  if [ -z "$format" ]; then
    format='%.8f'
  fi

  echo "$(utils::normalize_expr "$expression" | bc -l | xargs printf "$format")"

  return 0
}

# Halt execution until a certain file/directory appears
# Arguments:
#   file: the file/directory to wait for
#   timeout: [optional] how long to wait before raising error. Default is 60s.
#                       Resolution for this is only seconds.
#   period: [optional] how frequently to check. Default is 1s.
function utils::wait_for_file {
  file="$1"
  timeout="$2"
  period="$3" # may be empty
  if [ -z "$timeout" ]; then
    timeout=60
  fi
  if [ -z "$period" ]; then
    period=1
  fi

  starttime=$(date +%s)

  while [ ! -e "$file" ]; do
    sleep "$period"

    currenttime=$(date +%s)
    if [ $((currenttime-starttime)) -gt $timeout ]; then
      utils::printerr '[Error] utils::wait_for_file: TIMEOUT'
      return 1
    fi
  done

  return 0
}

# Initialize a log file for some command with some useful information
# Arguments:
#   file: the log file to initialize
#   command: the command that writes to this logfile
function utils::init_logfile {
  file="$1"
  cmd="$2"
  if [ -f "$file" ]; then
    utils::printerr "[Warning] utils::init_logfile: $file already exists. Overwriting."
  fi

  echo "command=$cmd" > $file
  echo "[timing] START: $(date)" >> $file
  echo "SLURM_JOB_ID: $SLURM_JOB_ID" >> $file
  echo "SLURM_JOB_NAME: $SLURM_JOB_NAME" >> $file
  echo "SLURM_PROCID: $SLURM_PROCID" >> $file
  echo "startseconds=$(date +%s)" >> $file

  return 0
}

# Finalize a log file with some useful information
# Arguments:
#   file: the log file to finalize
function utils::finish_logfile {
  file="$1"
  if [ ! -f "$file" ]; then
    utils::printerr "[Warning] utils::finish_logfile: $file does not exist. Not appending."
    return 0
  fi

  endseconds=$(date +%s)
  startseconds=$(grep -m 1 -oP "startseconds=+\K\d*" $file)
  diffseconds=$((endseconds - startseconds))
  diffhuman="$(printf '%d:%.2d:%.2d' $((diffseconds/3600)) $((diffseconds % 3600 / 60)) $((diffseconds % 60)))"

  echo "endseconds=$endseconds" >> $file
  echo "[timing] END: $(date)" >> $file
  echo "[timing] Took seconds=$diffseconds" >> $file
  echo "[timing] Took $diffhuman" >> $file

  return 0
}

# Run a command, redirecting output to logfile
# Arguments:
#   command: the command to be executed
#   logfile: the logfile
function utils::run {
  cmd="$1"
  logfile="$2"

  echo "[timing] $(date) starting $cmd"
  utils::init_logfile "$logfile" "$cmd"

  # some trickery to get around set -e
  eval "$cmd >> $logfile" && status=$? || status=$?

  if [ "$status" -ne "0" ]; then
    utils::printerr "[Error] exit code: $status; failed command: $cmd"
    return 1
  fi

  utils::finish_logfile "$logfile"
  echo "[timing] $(date) finished $cmd"

  return 0
}
