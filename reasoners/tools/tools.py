from reasoners import Tool
import reasoners.tools.wikiutils as wikiutils

wikisearch = Tool(
        func=wikiutils.webthink,
        name="wikisearch",
        description="Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search."
    )

wikilookup = Tool(
        func=wikiutils.webthink,
        name="wikilookup",
        description="Lookup[keyword], which returns the next sentence containing keyword in the current passage."
    )

wikifinish = Tool(
        func=wikiutils.webthink,
        name="finish",
        description="Finish[answer], which returns the answer and finishes the task."
    )