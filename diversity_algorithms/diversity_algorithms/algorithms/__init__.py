 # coding: utf-8

from diversity_algorithms.algorithms.novelty_search import novelty_ea
# from diversity_algorithms.algorithms.ea import ea
# No reason to expose other functions - they can be accessed through the submodule if needed


__all__=["novelty_search", "ea", "stats", "utils", "novelty_management", "quality_diversity", "evolutionary_algorithms"]
