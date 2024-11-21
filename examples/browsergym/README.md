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
# download nltk punkt_tab
python -c "import nltk; nltk.download('punkt_tab')"

# set up webarena environment variables
export BASE_URL="http://localhost"
export WA_SHOPPING="$BASE_URL:7770/"
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

Then serve the webarena server following the [webarena instructions](https://github.com/web-arena-x/webarena/blob/main/environment_docker/README.md).
After that, you should be able to run
```bash
python wa_test.py
```
to run a webarena task.

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
