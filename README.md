
# README 

### Notes:
1. The pretrained model for CIFAR-10 can be downloaded from [this repo](https://github.com/cmhcbb/attackbox).

This is for releasing the source code of the paper "Februus: Input Purification Defense Against Trojan Attacks on Deep Neural Network Systems" 

Archived Version: [RamBoAttack](https://arxiv.org/abs/2112.05282)

The project is published as part of the following paper and if you re-use our work, please cite the following paper:


```
@inproceedings{vo2022,
title={RamBoAttack: A Robust Query Efficient Deep Neural Network Decision Exploit},
author={Viet Quoc Vo and Ehsan Abbasnejad and Damith C. Ranasinghe},
year = {2022},
journal = {Network and Distributed Systems Security (NDSS) Symposium},
}
```

The source code is written mostly on *Python 3* and *Pytorch*, so please help to download and install Python3 and Pytorch beforehand.

# Requirements

To install the requirements for this repo, run the following command: 
```
git clone https://github.com/RamBoAttack/RamBoAttack.github.io.git
cd RamBoAttack
pip3 install -r requirements.txt
```

# Run the RamBoAttack

There are two ways to run the method:

- The first way is to run step-by-step with the Jupyter Notebook file *RamBoAttack.ipynb* in the main folder. 

- The second way is to run the *test.py* file. This is to run the RamBoAttack on the whole test set for that task.: 

```python
# For example, to run RamBoAttack on CIFAR10
cd main
python3 test.py
```
  
## TODO 
- [ ] add the testing code

