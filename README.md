# Difference Residual Graph Neural Networks

This repository contains a PyTorch implementation of ["Difference Residual Graph Neural Networks"](https://yangliang.github.io/pdf/mm22.pdf).

## Runtime Environment

* CUDA=11.1

```
conda env create -f environment.yml
```

## Run All Benchmarks

```
cd scripts
chmod 744 run_all.sh
./run_all.sh
```

## Citation

```
@article{yangMM2022drc,
  title = {Difference Residual Graph Neural Networks},
  author = {Liang Yang, Weihang Peng and Wenmiao Zhou, Bingxin Niu and Junhua Gu, Chuan Wang and Yuanfang Guo, Dongxiao He and Xiaochun Cao},
  year = {2022},
  booktitle = {{MM} '22: The {ACM} MULTIMEDIA Conference 2022},
}
```