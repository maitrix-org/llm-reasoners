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

## Webarena - locally hosted

Note: There seems to be a bug in browsergym/playwright which makes using demo_mode on webarena infinitely stall the browser for fill() actions. Be sure to set demo_mode to off in the HighlevelActionSet.

```bash

export BASE_URL="http://localhost"
export WA_SHOPPING="$BASE_URL:7770/"
export WA_SHOPPING_ADMIN="$BASE_URL:7780"
export WA_REDDIT="$BASE_URL:9999"
export WA_GITLAB="$BASE_URL:8023"
# using 8898 here due to lab server having 8888 in use. change to whatever port configured for webarena wikipedia
export WA_WIKIPEDIA="$BASE_URL:8898/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:3000"
export WA_HOMEPAGE="$BASE_URL:4399"

```

## NOTES

Since browsergym relies on an external environment for state, it doesn't make sense to try to textually represent the environment state. For the "example" passed into the Reasoner object, it's just empty, and instead it's the environment configuration with the "name" attribute that loads in the example.
