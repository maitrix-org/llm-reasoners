# Development notes

## Developer Setup
### 1. Set up submodule
```
cd examples/
git submodule add git@github.com:BlankCheng/AgentLab.git AgentLab
pip install -e .
```
### 2. Install browsergym
```
git submodule add https://github.com/ServiceNow/BrowserGym.git BrowserGym
cd BrowserGym
make install
```
### 3. Set up datasets
Refer to [README.md](./README.md)