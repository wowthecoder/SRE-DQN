# SRE-DQN
Deep Reinforcement Learning for Strategically Robust Equilibria

cp .env.example .env
# then set PATH_LICENSE_STRING in .env

cd discrete_action_space

export LD_LIBRARY_PATH="$PWD/pathlib/lib_lnx:$LD_LIBRARY_PATH"

gcc -shared -fPIC -Ipathlib/include -Ipathlib/examples/C -o pathwrap.so \
    pathwrap.c pathlib/examples/C/Persistent_options.c \
    -Lpathlib/lib_lnx -lpath50 -lm -ldl \
    -Wl,-rpath,'$ORIGIN/pathlib/lib_lnx'

Virtual environment managed in conda, python version is 3.9 because marllib cannot go along with newer Python versions.
Stored in /vol/bitbucket/jhl323/miniconda3/envs/fypenv
Do `conda activate fypenv` to activate venv
