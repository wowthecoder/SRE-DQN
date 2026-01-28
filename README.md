# SRE-DQN
Deep Reinforcement Learning for Strategically Robust Equilibria

export PATH_LICENSE_STRING="1259252040&Courtesy&&&USR&GEN2035&5_1_2026&1000&PATH&GEN&31_12_2035&0_0_0&6000&0_0"

export LD_LIBRARY_PATH="$PWD/pathlib/lib_lnx:$LD_LIBRARY_PATH"

gcc -shared -fPIC -Ipathlib/include -Ipathlib/examples/C -o pathwrap.so \
    pathwrap.c pathlib/examples/C/Standalone_Path.c \
    -Lpathlib/lib_lnx -lpath50 -lm -ldl \
    -Wl,-rpath,'$ORIGIN/pathlib/lib_lnx'
