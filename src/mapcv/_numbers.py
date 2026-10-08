"""A base class for the config models that refuses YAML booleans where a number belongs.

YAML 1.1 reads ``yes``, ``on`` and ``true`` as booleans, and pydantic's lax mode turns a
boolean into ``1`` (or ``1.0``) for an ``int`` or ``float`` field, so ``zoom: yes`` or
``patch_size: true`` would pass as 1. Integers stay valid for float fields.
"""

from __future__ import annotations

import types
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, model_validator

__all__ = ["NoBooleanNumbers"]

_NUMBERS = (int, float)


def _leaves(annotation: Any) -> set[Any]:
    """The plain types an annotation is made of (``int | None`` gives ``{int, NoneType}``)."""
    origin = get_origin(annotation)
    if origin is Annotated:
        return _leaves(get_args(annotation)[0])
    if origin is Literal:
        return {type(value) for value in get_args(annotation)}
    if origin in (Union, types.UnionType, list, tuple, set, frozenset):
        found: set[Any] = set()
        for arg in get_args(annotation):
            found |= _leaves(arg)
        return found
    return {annotation}


def _holds_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (list, tuple)):
        return any(_holds_boolean(item) for item in value)
    return False


class NoBooleanNumbers(BaseModel):
    """A model whose number fields (``int``, ``float``) reject ``true`` and ``false``."""

    # A model validator, not a field validator on "*": pydantic refuses a field validator
    # on the discriminator field of the imagery and labels unions.
    @model_validator(mode="before")
    @classmethod
    def _numbers_are_not_booleans(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        for name, field in cls.model_fields.items():
            value = data.get(name, data.get(field.alias or name))
            if not _holds_boolean(value):
                continue
            kinds = _leaves(field.annotation)
            if kinds & {bool, Any, object} or not any(kind in _NUMBERS for kind in kinds):
                continue
            raise ValueError(
                f"{name}: {str(value).lower()} is a YAML boolean, not a number; write the "
                "number itself (YAML reads yes, on and true as booleans)"
            )
        return data
