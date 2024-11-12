# Browsergym

https://github.com/ServiceNow/BrowserGym

## Setting up browsergym

```bash
cd ./examples/browsergym

git clone https://github.com/ServiceNow/BrowserGym.git
cd BrowserGym
make install
```

## Installing a dataset - miniwob++

Can choose an arbitrary dataset to install, but miniwob is the easiest to setup.
For other datasets, go to the browsergym readme, and follow the instructions there.

```bash
cd ./examples/browsergym
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git

cd miniwob-plusplus/miniwob/html/miniwob
export MINIWOB_URL="file://$(pwd)/"
```
