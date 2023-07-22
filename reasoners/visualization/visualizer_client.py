import dataclasses
import json
from typing import Optional

import requests

from reasoners.visualization import TreeLog, TreeSnapshot

_API_DEFAULT_BASE_URL = "https://2wz3t0av30.execute-api.us-west-1.amazonaws.com/staging"
_VISUALIZER_DEFAULT_BASE_URL = "https://www.llm-reasoners.net"


class VisualizerClient:
    def __init__(self, base_url: str = _API_DEFAULT_BASE_URL) -> None:
        self.base_url = base_url

    @dataclasses.dataclass
    class TreeLogReceipt:
        id: str
        access_key: str

        def access_url(self) -> str:
            return f"{_VISUALIZER_DEFAULT_BASE_URL}/visualizer/{self.id}?accessKey={self.access_key}"

    class _TreeLogEncoder(json.JSONEncoder):
        def default(self, o):
            from numpy import float32

            if isinstance(o, TreeSnapshot.Node):
                return o.__dict__
            elif isinstance(o, TreeSnapshot.Edge):
                return o.__dict__
            elif isinstance(o, TreeSnapshot):
                return o.__dict__()
            elif isinstance(o, float32):
                return float(o)
            if isinstance(o, TreeLog):
                return {"logs": list(o)}
            else:
                return super().default(o)

    def post_log(self, data: TreeLog) -> Optional[TreeLogReceipt]:
        url = f"{self.base_url}/logs"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(data, cls=self._TreeLogEncoder))

        if response.status_code != 200:
            print(f"POST Log failed with status code: {response.status_code}, message: {response.text}")
            return None

        return self.TreeLogReceipt(**response.json())
