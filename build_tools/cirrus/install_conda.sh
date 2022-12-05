curl -L -o mambaforge.sh $MINICONDA_URL

MINICONDA_PATH=$HOME/miniconda
chmod +x mambaforge.sh && ./mambaforge.sh -b -p $MINICONDA_PATH
export PATH=$MINICONDA_PATH/bin:$PATH

conda init --all --verbose
conda update --yes conda
echo $CONDA_PREFIX
