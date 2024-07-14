#!/usr/bin/env bash
#
# Start an instance of the mardi-ai-engine docker container.
# 

show_help() {
    echo " "
    echo "usage: Starts the Docker container and runs a user-specified command"
    echo " "
    echo "   ./docker/run.sh --run RUN_COMMAND"
    echo " "
    echo "args:"
    echo " "
    echo "   --help                       Show this help text and quit"
    echo " "
    echo "   -r, --run RUN_COMMAND  Command to run once the container is started."
    echo "                          Note that this argument must be invoked last,"
    echo "                          as all further arguments will form the command."
    echo "                          If no run command is specified, an interactive"
    echo "                          terminal into the container will be provided."
}

die() {
    printf '%s\n' "$1"
    show_help
    exit 1
}

# paths to some project directories
NOTEBOOK_DIR="notebook"
IMAGES_DIR="images"

DOCKER_ROOT="/data"	# where the project resides inside docker

# generate mount commands
DATA_VOLUME=" \
--volume $PWD/$NOTEBOOK_DIR:$DOCKER_ROOT/$NOTEBOOK_DIR \
--volume $PWD/$IMAGES_DIR:$DOCKER_ROOT/$IMAGES_DIR "

# parse user arguments
USER_COMMAND=""

while :; do
    case $1 in
        -h|-\?|--help)
            show_help
            exit
            ;;
        -r|--run)
            if [ "$2" ]; then
                shift
                USER_COMMAND=" $@ "
            else
                die 'ERROR: "--run" requires a non-empty option argument.'
            fi
            ;;
        --)
            shift
            break
            ;;
        -?*)
            printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
            ;;
        *)   # default case: No more options, so break out of the loop.
            break
    esac

    shift
done

# select container tag (unless specified by user)
CONTAINER_IMAGE="pintarai/mardi:latest"

# check for V4L2 devices
V4L2_DEVICES=""

for i in {0..9}
do
	if [ -a "/dev/video$i" ]; then
		V4L2_DEVICES="$V4L2_DEVICES --device /dev/video$i "
	fi
done

# check for display
DISPLAY_DEVICE=""

if [ -n "$DISPLAY" ]; then
	sudo xhost +si:localuser:root
	DISPLAY_DEVICE=" -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix "
fi

# print configuration
print_var() 
{
	if [ -n "${!1}" ]; then                                                # reference var by name - https://stackoverflow.com/a/47768983
		local trimmed="$(echo -e "${!1}" | sed -e 's/^[[:space:]]*//')"   # remove leading whitespace - https://stackoverflow.com/a/3232433    
		printf '%-17s %s\n' "$1:" "$trimmed"                              # justify prefix - https://unix.stackexchange.com/a/354094
	fi
}

print_var "CONTAINER_IMAGE"
print_var "DATA_VOLUME"
print_var "USER_COMMAND"
print_var "V4L2_DEVICES"
print_var "DISPLAY_DEVICE"

# run the container
sudo docker run --gpus all -it --rm \
		--network=host \
		--shm-size=8g \
		--ulimit memlock=-1 \
		--ulimit stack=67108864 \
		-e NVIDIA_DRIVER_CAPABILITIES=all \
		-w $DOCKER_ROOT \
		$DISPLAY_DEVICE $V4L2_DEVICES \
		$DATA_VOLUME $USER_VOLUME $DEV_VOLUME \
		$CONTAINER_IMAGE $USER_COMMAND

