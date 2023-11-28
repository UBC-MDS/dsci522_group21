FROM quay.io/jupyter/minimal-notebook:2023-11-19

RUN conda install -y \
    matplotlib=3.8.2 \
    pandas=2.1.3 \
    scikit-learn=1.3.2 \
    altair=5.1.2 \
    pytest=7.4.3 \
    dataframe_image=0.1.1 \
    click=8.1.7

RUN pip install "vegafusion[embed]>=1.4.0"
