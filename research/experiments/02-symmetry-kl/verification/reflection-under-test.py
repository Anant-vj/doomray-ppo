"""Training-only left/right reflection for the official structured observation."""

import numpy as np

from doom_env.spaces import BUTTONS, ENTITY_FEATURES, MAX_ENTITIES, NUM_RAYS, VITALS_FIELDS
from submission.policy import HEADS

# Ray 0 faces forward and positive ray indices rotate toward the player's left.
RAY_REFLECTION = tuple((-index) % NUM_RAYS for index in range(NUM_RAYS))
BUTTON_REFLECTION = (0, 1, 3, 2, 5, 4, 6, 7, 8)
_DIRECTION_HEADS = (1, 2)
_HEAD_STARTS = np.cumsum((0, *HEADS[:-1])).tolist()
LOGIT_REFLECTION = tuple(
    start + label
    for head, (start, size) in enumerate(zip(_HEAD_STARTS, HEADS, strict=True))
    for label in ((0, 2, 1) if head in _DIRECTION_HEADS else range(size))
)

_ENTITY_WIDTH = len(ENTITY_FEATURES)
_ENTITY_SIGNS = tuple(ENTITY_FEATURES.index(name) for name in ("sin_rel_angle", "rel_vel_right"))
_RIGHT_VITAL = VITALS_FIELDS.index("vel_right")

# The offsets follow submission.policy.pack, including its flattened entity rows.
_ENTITIES_START = 2 * NUM_RAYS
_ENTITIES_END = _ENTITIES_START + MAX_ENTITIES * _ENTITY_WIDTH
_MASK_START = _ENTITIES_END + MAX_ENTITIES
_VITALS_START = _MASK_START + MAX_ENTITIES
_BUTTONS_START = _VITALS_START + len(VITALS_FIELDS)
_PACKED_REFLECTION = (
    *RAY_REFLECTION,
    *(NUM_RAYS + index for index in RAY_REFLECTION),
    *range(_ENTITIES_START, _BUTTONS_START),
    *(_BUTTONS_START + index for index in BUTTON_REFLECTION),
)


def _last_dim(values, expected):
    if not hasattr(values, "shape"):
        values = np.asarray(values)
    if not values.ndim or values.shape[-1] != expected:
        raise ValueError(f"Expected a final dimension of {expected}, got {values.shape}.")
    return values


def reflect_observation(obs):
    """Reflect the structured input. Carry the policy's unused pixel fields through."""
    rays = obs["rays"]
    entities = obs["visible_entities"]
    features = entities["features"].copy()
    mask = entities["mask"]
    features[..., _ENTITY_SIGNS] *= np.where(mask[..., None], -1, 1)
    vitals = obs["vitals"].copy()
    vitals[..., _RIGHT_VITAL] *= -1
    return {
        **obs,
        "rays": {
            "distance": rays["distance"][..., RAY_REFLECTION],
            "hit_type": rays["hit_type"][..., RAY_REFLECTION],
        },
        "visible_entities": {
            "features": features,
            "entity_type": entities["entity_type"].copy(),
            "mask": mask.copy(),
        },
        "vitals": vitals,
    }


def reflect_buttons(buttons):
    """Swap strafe-left/right and turn-left/right in NumPy or Torch actions."""
    return _last_dim(buttons, len(BUTTONS))[..., BUTTON_REFLECTION]


def reflect_choices(choices):
    """Reflect the six categorical labels, preserving a neutral direction."""
    choices = _last_dim(choices, len(HEADS))
    result = choices.copy() if isinstance(choices, np.ndarray) else choices.clone()
    directions = choices[..., _DIRECTION_HEADS]
    result[..., _DIRECTION_HEADS] = (3 - directions) * (directions != 0)
    return result


def reflect_logits(logits):
    """Align flat categorical logits with the opposite observation orientation."""
    return _last_dim(logits, sum(HEADS))[..., LOGIT_REFLECTION]


def reflect_packed(packed):
    """Reflect packed policy inputs, including previous buttons, frame by frame."""
    packed = _last_dim(packed, len(_PACKED_REFLECTION))
    result = packed[..., _PACKED_REFLECTION]
    entity_sign = 1 - 2 * packed[..., _MASK_START:_VITALS_START]
    for feature in _ENTITY_SIGNS:
        result[..., _ENTITIES_START + feature:_ENTITIES_END:_ENTITY_WIDTH] *= entity_sign
    result[..., _VITALS_START + _RIGHT_VITAL] *= -1
    return result
