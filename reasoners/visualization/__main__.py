def main():
    import argparse

    from reasoners.visualization import VisualizerClient

    parser = argparse.ArgumentParser()
    parser.add_argument('tree_log', type=str)
    parser.add_argument('--base_url', type=str)
    args = parser.parse_args()

    if args.base_url is None:
        client = VisualizerClient()
    else:
        client = VisualizerClient(args.base_url)

    with open(args.tree_log) as f:
        data = f.read()
    result = client.post_log(data)
    print(result.access_url)
