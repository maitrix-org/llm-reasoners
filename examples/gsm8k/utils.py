def retrieve_answer(output: str) -> Optional[str]:
    match = re.match(r'.*The answer is .*?([ $.0-9,\-]+).*\.$', result)
    if match is None:
        continue
    sub_answer = match[1].replace(',', '').replace('$', '').replace(' ', '')