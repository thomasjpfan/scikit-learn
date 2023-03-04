FROM mcr.microsoft.com/devcontainers/miniconda

USER vscode
COPY build_tools/circle/doc_environment.yml /home/vscode/env.yml

RUN conda install -c conda-forge mamba -y
RUN mamba env create --file /home/vscode/env.yml --name sklearn-dev && \
    mamba clean --all

RUN conda init bash && \
    echo "source activate sklearn-dev" >> ~/.bashrc && \
    echo "export SKLEARN_BUILD_PARALLEL=\$(python -c \"import joblib; print(joblib.cpu_count())\")" >> ~/.bashrc
