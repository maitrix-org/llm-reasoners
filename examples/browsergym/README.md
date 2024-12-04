# Browsergym

https://github.com/ServiceNow/BrowserGym

## Setting up browsergym

```bash
cd ./examples/browsergym

git clone https://github.com/ServiceNow/BrowserGym.git
cd BrowserGym
make install
```

## Installing datasets
### 1. Miniwob++

Can choose an arbitrary dataset to install, but miniwob is the easiest to setup.
For other datasets, go to the browsergym readme, and follow the instructions there.

```bash
cd ./examples/browsergym
git clone https://github.com/Farama-Foundation/miniwob-plusplus.git

cd miniwob-plusplus/miniwob/html/miniwob
export MINIWOB_URL="file://$(pwd)/"
```

### 2. Webarena - locally hosted

Note: There seems to be a bug in browsergym/playwright which makes using demo_mode on webarena infinitely stall the browser for fill() actions. Be sure to set demo_mode to off in the HighlevelActionSet.

```bash
# download nltk punkt_tab
python -c "import nltk; nltk.download('punkt_tab')"

# set up webarena environment variables
export BASE_URL="http://localhost"
export WA_SHOPPING="$BASE_URL:7770"
export WA_SHOPPING_ADMIN="$BASE_URL:7780"
export WA_REDDIT="$BASE_URL:9999"
export WA_GITLAB="$BASE_URL:8023"
# using 8898 here due to lab server having 8888 in use. change to whatever port configured for webarena wikipedia
export WA_WIKIPEDIA="$BASE_URL:8898/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:3000"
export WA_HOMEPAGE="$BASE_URL:4399"

# set up openai api key
export OPENAI_API_KEY=...
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

### 3. VisualWebArena -- locally hosted
```bash
# visualwebarena environment variables (change ports as needed)
export VWA_CLASSIFIEDS="$BASE_URL:8083"
export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
export VWA_SHOPPING="$BASE_URL:8082"
export VWA_REDDIT="$BASE_URL:8080"
export VWA_WIKIPEDIA="$BASE_URL:8081"
export VWA_HOMEPAGE="$BASE_URL:80"

# if your webarena instances offers the FULL_RESET feature (optional)
export VWA_FULL_RESET="$BASE_URL:7565"
```

Then serve the visualwebarena server following the [visualwebarena instructions](https://github.com/web-arena-x/visualwebarena/blob/main/environment_docker/README.md). e.g., for the reddit instance (note the .tar file needs downloading), run
```bash
docker load --input postmill-populated-exposed-withimg.tar
docker run --name forum -p 8080:80 -d postmill-populated-exposed-withimg
```

Test the reddit instance by running
```bash
curl $VWA_REDDIT
```

After that, you should be able to run
```bash
python vwa_test.py
```
to run a visualwebarena task.

## Visualize search tree

```bash
python visualize.py --tree_log_file=<path_to_tree_log_pickle_file>
```

## Troubleshooting
If you meet the error:
```
ModuleNotFoundError: No module named 'huggingface_hub.errors'
```
That's due to a conflict of huggingface-hub version. You can fix it by running:
```bash
pip install huggingface_hub==0.24.7
```

If you meet the error:
```
playwright._impl._errors.TargetClosedError: BrowserType.launch: Target page, context or browser has been closed Browser logs: ╔════════════════════════════════════════════════════════════════════════════════════════════════╗ ║ Looks like you launched a headed browser without having a XServer running. ║ ║ Set either 'headless: true' or use 'xvfb-run <your-playwright-app>' before running Playwright. ║ ║ ║ ║ <3 Playwright Team
```
Try set `headless=True` in the `get_env` function in `support.py`. This usually happens if you run the code in a remote server without a display.

## NOTES

Since browsergym relies on an external environment for state, it doesn't make sense to try to textually represent the environment state. For the "example" passed into the Reasoner object, it's just empty, and instead it's the environment configuration with the "name" attribute that loads in the example.
