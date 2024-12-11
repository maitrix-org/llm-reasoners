# browsergym

https://github.com/ServiceNow/BrowserGym

## Overview

## Setting up browsergym

```bash
git clone https://github.com/ServiceNow/BrowserGym.git
cd BrowserGym
make install
```

## Installing datasets

### 1. Miniwob++

https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/miniwob/README.md

```bash
cd ./examples/browsergym
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git

cd miniwob-plusplus/miniwob/html/miniwob
export MINIWOB_URL="file://$(pwd)/"
```

### 2. Webarena - Locally Hosted

https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/webarena/README.md

```bash
# download nltk punkt_tab
python -c "import nltk; nltk.download('punkt_tab')"

# set up webarena environment variables
export BASE_URL="http://localhost"
export WA_SHOPPING="$BASE_URL:7770"
export WA_SHOPPING_ADMIN="$BASE_URL:7780"
export WA_REDDIT="$BASE_URL:9999"
export WA_GITLAB="$BASE_URL:8023"
export WA_WIKIPEDIA="$BASE_URL:8898/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:3000"
export WA_HOMEPAGE="$BASE_URL:4399"

```

Then serve the webarena server following the [webarena instructions](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md). e.g., for the shopping instance (note the .tar file needs downloading), run

```bash
docker load --input shopping_final_0712.tar
docker run --name shopping -p 7770:80 -d shopping_final_0712
# wait ~1 min to wait all services to start

docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="${WA_SHOPPING}" # no trailing slash
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e  'UPDATE core_config_data SET value="${WA_SHOPPING}" WHERE path = "web/secure/base_url";'
docker exec shopping /var/www/magento2/bin/magento cache:flush
```

Test the shopping instance by running

```bash
curl $WA_SHOPPING
```

After that, you should be able to run

```bash
python wa_test.py
```

to run a webarena task.

## API Keys

```bash
# i.e. openai
export OPENAI_API_KEY="your_openai_api_key"
```

## Running Reasoning-as-Planning (RAP) on a Task

```bash

python inference.py <task_name>

```

## Visualize Search Tree

Some MCTSResult pickle objects have been provided in the results/tree-search directory.

```bash
# python visualize.py <task_name>
python visualize.py webarena.599
```
