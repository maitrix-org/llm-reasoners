import dataclasses
import json
import gzip
from typing import Optional, Union

import requests

from reasoners.algorithm import MCTSResult, BeamSearchResult, DFSResult
from reasoners.visualization import TreeLog, TreeLogEncoder

_API_DEFAULT_BASE_URL = "https://4lgdwukvng.execute-api.us-east-1.amazonaws.com/main"
_VISUALIZER_DEFAULT_BASE_URL = "https://main.d1puk3wdon4rk8.amplifyapp.com"


class VisualizerClient:
    def __init__(self, base_url: str = _API_DEFAULT_BASE_URL) -> None:
        self.base_url = base_url

    @dataclasses.dataclass
    class UploadUrl:
        upload_url: dict
        file_name: dict

    @dataclasses.dataclass
    class TreeLogReceipt:
        id: str
        access_key: str

        @property
        def access_url(self) -> str:
            return f"{_VISUALIZER_DEFAULT_BASE_URL}/visualizer/{self.id}?accessKey={self.access_key}"

    def get_upload_url(self) -> Optional[UploadUrl]:
        print("Getting log upload link...")
        url = f"{self.base_url}/logs/get-upload-url"
        response = requests.get(url)
        if response.status_code != 200:
            print(
                f"GET Upload URL failed with status code: {response.status_code}, message: {response.text}"
            )
            return None
        return self.UploadUrl(**response.json())

    def post_log(
        self, data: Union[TreeLog, str, dict], upload_url: UploadUrl
    ) -> Optional[TreeLogReceipt]:
        if isinstance(data, TreeLog):
            data = json.dumps(data, cls=TreeLogEncoder)
        if isinstance(data, dict):
            data = json.dumps(data)

        print(f"Tree log size: {len(data)} bytes")
        data = gzip.compress(data.encode("utf-8"))
        files = {"file": (upload_url.file_name, data)}

        print(f"Tree log compressed size: {len(data)} bytes")
        print("Uploading log...")
        response = requests.post(
            upload_url.upload_url["url"],
            data=upload_url.upload_url["fields"],
            files=files,
        )

        if response.status_code != 200 and response.status_code != 204:
            print(
                f"POST Log failed with status code: {response.status_code}, message: {response.text}"
            )
            return None

        response = requests.post(
            f"{self.base_url}/logs/upload-complete",
            json={"file_name": upload_url.file_name},
        )

        if response.status_code != 200:
            print(
                f"POST Upload Complete failed with status code: {response.status_code}, message: {response.text}"

        return self.TreeLogReceipt(**response.json())


def present_visualizer(receipt: VisualizerClient.TreeLogReceipt):
    import webbrowser

    print(f"Visualizer URL: {receipt.access_url}")
    webbrowser.open(receipt.access_url)


def visualize(
    result: Union[TreeLog, MCTSResult, BeamSearchResult, DFSResult], **kwargs
):
    tree_log: TreeLog

    if isinstance(result, TreeLog):
        tree_log = result
    elif isinstance(result, MCTSResult):
        tree_log = TreeLog.from_mcts_results(result, **kwargs)
    elif isinstance(result, BeamSearchResult):
        tree_log = TreeLog.from_beam_search_results(result, **kwargs)
    elif isinstance(result, DFSResult):
        tree_log = TreeLog.from_dfs_results(result, **kwargs)
    elif isinstance(result, ...):
        raise NotImplementedError()
    else:
        raise TypeError(f"Unsupported result type: {type(result)}")

    client = VisualizerClient()
    upload_url = client.get_upload_url()
    receipt = client.post_log(tree_log, upload_url)
