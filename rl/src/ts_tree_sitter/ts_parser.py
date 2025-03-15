from tree_sitter import Parser
from ts_tree_sitter.ts_language import GoLanguage, CppLanguage


class GoParser:
    def __init__(self):
        self.parser = Parser(GoLanguage.language)

    def parse(self, code):
        tree = self.parser.parse(bytes(code, 'utf8'))
        return tree
    

class CppParser:
    def __init__(self):
        self.parser = Parser(CppLanguage.language)

    def parse(self, code):
        tree = self.parser.parse(bytes(code, 'utf8'))
        return tree

