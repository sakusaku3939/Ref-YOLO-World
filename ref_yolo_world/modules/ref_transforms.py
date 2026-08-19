"""Transforms for referring expression training."""

import re

from mmyolo.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SpatialTextFlipCorrection:
    """Synchronize left/right words with horizontal image flips."""

    _SPATIAL_PATTERN = re.compile(
        r'\b(leftmost|rightmost|left|right)\b', flags=re.IGNORECASE)
    _SPATIAL_SWAP = {
        'left': 'right',
        'right': 'left',
        'leftmost': 'rightmost',
        'rightmost': 'leftmost',
    }

    def __call__(self, results: dict) -> dict:
        if not results.get('flip') or results.get('flip_direction') != 'horizontal':
            return results

        if 'texts' in results:
            results['texts'] = self._replace_nested(results['texts'])
        return results

    def _replace_nested(self, value):
        if isinstance(value, str):
            return self._replace_text(value)
        if isinstance(value, list):
            return [self._replace_nested(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._replace_nested(item) for item in value)
        return value

    def _replace_text(self, text: str) -> str:
        return self._SPATIAL_PATTERN.sub(self._swap_match, text)

    def _swap_match(self, match: re.Match) -> str:
        original = match.group(0)
        swapped = self._SPATIAL_SWAP[original.lower()]
        if original.isupper():
            return swapped.upper()
        if original[0].isupper():
            return swapped.capitalize()
        return swapped
