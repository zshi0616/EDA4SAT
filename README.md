# EDA-Driven Preprocessing for SAT Solving

Authors: Zhengyuan Shi, Tiebing Tang, Sadaf Khan, Hui-Ling Zhen, Mingxuan Yuan, Zhufei Chu and Qiang Xu

## Abstract 
Effective formulation of problems into Conjunctive Normal Form (CNF) is critical in modern SAT (Boolean Satisfiability) solving for optimizing solver performance. Addressing the limitations of existing methods, our EDA-driven preprocessing framework introduces a novel methodology for preparing SAT problems, leveraging both circuit and CNF formats for enhanced flexibility and efficiency. Central to our approach is the integration of a new logic synthesis technique, guided by a reinforcement learning agent, and a novel cost-customized LUT mapping strategy, enabling efficient handling of diverse SAT challenges. Our framework demonstrates substantial performance improvements, evidenced by a 96.14% reduction in average solving time for a set of circuit-based logic equivalence checking problems and a 52.42% reduction for non-circuit SAT problem instances, compared to applying CNF-based solver directly. 

## Installation 
1. See [abc official repo](https://github.com/berkeley-abc/abc) to install abc

2. See [DeepGate official repo](https://github.com/Ironprop-Stone/python-deepgate) to install python-deepgate

3. Install mockturtle (cost-customized mapper) and kissat (baseline solver)
```sh
bash install.sh 
```

## Train
Train your own RL agent with training dataset. Please copy the `.aiger` format training samples to folder `dataset/train`. We also provide three training samples reported in our papers. 

To train the RL agent, run the following scripe. The trained network will be stored in `./exp/train/qnet_last.pth`
```sh 
bash run/train.sh
```

## Test 
We provide same testing samples reported in Section 4.1 (I1-I5). These samples are collected from industrial logic equvalence checking (LEC) problems. Run the following scripe to test our framework:
```sh
bash run/test.sh
```

