FROM quay.io/jupyter/minimal-notebook:2023-11-19

COPY environment.yml environment.yml

RUN conda env update --file environment.yml --prune