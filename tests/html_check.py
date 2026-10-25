"""Checks that a piece of HTML closes every tag it opens, in order.

test_render.py and test_guide.py both use it. It lives here rather than in test_render.py, which
imports torch through the harness's backends, so that test_guide.py runs with pygments and pytest alone,
as the Pages workflow runs it before building the guide.
"""

from html.parser import HTMLParser

VOID = {"br", "img", "hr", "input", "meta", "link", "wbr"}


class TagChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self.stack, self.problems = [], []

    def handle_starttag(self, tag, attrs):
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"</{tag}> while open: {self.stack}")
        else:
            self.stack.pop()


def assert_balanced(markup: str):
    checker = TagChecker()
    checker.feed(markup)
    checker.close()
    assert checker.problems == [] and checker.stack == [], (checker.problems, checker.stack)
