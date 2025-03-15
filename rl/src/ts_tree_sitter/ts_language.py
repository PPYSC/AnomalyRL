import tree_sitter_go as tsgo
import tree_sitter_cpp as tscpp
from tree_sitter import Language


class GoLanguage:
    language = Language(tsgo.language())

    @staticmethod
    def use_query(query, node):
        return GoLanguage.language.query(query).captures(node)
    
    
class CppLanguage:
    language = Language(tscpp.language())

    @staticmethod
    def use_query(query, node):
        return CppLanguage.language.query(query).captures(node)
