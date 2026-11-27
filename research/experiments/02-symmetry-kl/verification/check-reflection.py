from itertools import product
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from doom_env import sensors
from submission.policy import HEADS, buttons, pack
from training.reflection import RAY_REFLECTION, reflect_observation, reflect_buttons, reflect_choices, reflect_packed, reflect_logits

torch.set_num_threads(1)
assert (RAY_REFLECTION[0], RAY_REFLECTION[1], RAY_REFLECTION[16], RAY_REFLECTION[32], RAY_REFLECTION[48], RAY_REFLECTION[63]) == (0, 63, 48, 32, 16, 1)
labels = np.array(list(product(*(range(n) for n in HEADS))), dtype=np.int64)
buttons_np = buttons(labels)
assert len(np.unique(reflect_buttons(buttons_np), axis=0)) == 216
np.testing.assert_array_equal(reflect_buttons(reflect_buttons(buttons_np)), buttons_np)
np.testing.assert_array_equal(reflect_choices(reflect_choices(labels)), labels)
np.testing.assert_array_equal(reflect_buttons(buttons_np), buttons(reflect_choices(labels)))
np.testing.assert_array_equal(reflect_buttons([1,0,0,0,0,0,1,1,1]), [1,0,0,0,0,0,1,1,1])
assert torch.equal(reflect_choices(torch.from_numpy(labels)), torch.from_numpy(reflect_choices(labels)))
assert torch.equal(reflect_buttons(torch.from_numpy(buttons_np)), torch.from_numpy(reflect_buttons(buttons_np)))
logits = torch.arange(2 * 4 * sum(HEADS)).reshape(2,4,-1)
aligned = reflect_logits(logits)
assert aligned[0,0].tolist() == [0,1,2,3,5,4,6,8,7,9,10,11,12,13,14]
assert torch.equal(reflect_logits(aligned), logits)
assert torch.equal(reflect_choices(logits.split(HEADS, -1)[0].new_tensor([0,2,1,0,1,1])), torch.tensor([0,1,2,0,1,1]))
root = Path(__file__).resolve().parents[2] / "01-behavior-cloning/demonstrations/human"
samples = [root/'KeyDoor/episode-0000/steps.npz', root/'BasicAim/episode-0000/steps.npz', root/'CenterCombat/episode-0000/steps.npz']
observations = windows = 0
for sample in samples:
    with np.load(sample, allow_pickle=False) as d:
        previous, accepted = d['previous_actions'], d['actions']
        recorded = d['packed']
        raw = dict(rays=dict(distance=d['ray_distance'],hit_type=d['ray_hit_type']),
                   visible_entities=dict(features=d['entity_features'],entity_type=d['entity_type'],mask=d['entity_mask']),
                   vitals=d['vitals'])
        np.testing.assert_array_equal(previous[1:], accepted[:-1])
        np.testing.assert_array_equal(previous[0], np.zeros(9))
        observed = reflect_observation(raw)
        recovered = reflect_observation(observed)
        np.testing.assert_array_equal(recovered['vitals'],raw['vitals'])
        for group in ('rays','visible_entities'):
            for key in raw[group]:
                assert recovered[group][key].tobytes() == raw[group][key].tobytes()
        black = raw['visible_entities']['mask'] == 0
        assert observed['visible_entities']['features'][black].tobytes() == raw['visible_entities']['features'][black].tobytes()
        packed_reflection = reflect_packed(recorded)
        assert reflect_packed(packed_reflection).tobytes() == recorded.tobytes()
        def row(i):
            return {'rays':{k:v[i] for k,v in observed['rays'].items()},
                    'visible_entities':{k:v[i] for k,v in observed['visible_entities'].items()},
                    'vitals':observed['vitals'][i]}
        to_pack=np.stack([pack(row(i),reflect_buttons(a)) for i,a in enumerate(previous)])
        np.testing.assert_array_equal(packed_reflection,to_pack)
        # Every complete four-frame sequence is independently reflected, retaining order.
        frames = np.stack([recorded[i:i+4] for i in range(len(recorded)-3)])
        expected = np.stack([to_pack[i:i+4] for i in range(len(recorded)-3)])
        assert torch.equal(reflect_packed(torch.from_numpy(frames)),torch.from_numpy(expected))
        observations += len(recorded)
        windows += len(frames)
# Physically mirror wall segments and a moving enemy in the official Sensors computation.
heading=.37
basis=np.array([np.cos(heading), np.sin(heading)])
transform=2*np.outer(basis,basis)-np.eye(2)
p1=np.array([[180., -150.],[-120., 90.]])
p2=np.array([[220., 60.],[38., 125.]])
enemy_xy=np.array([118.,72.])
velocity=np.array([-1.5,2.1])
player_vel=np.array([1.2,-.4])
def sense(matrix):
    geo=SimpleNamespace(p1=p1@matrix,p2=p2@matrix,move_block=np.ones(2,dtype=bool),
        sight_block=np.ones(2,dtype=bool),hit_type=np.array([sensors.RAY_WALL,sensors.RAY_LOCKED]),
        nearby=lambda x,y,rad:np.arange(2))
    pos=enemy_xy@matrix
    vx,vy=velocity@matrix
    actor=SimpleNamespace(name='Demon',category='Monster',id=7,position_x=pos[0],position_y=pos[1],velocity_x=vx,velocity_y=vy)
    rays,entities=sensors.Sensors(geo).compute(0,0,heading,tuple(player_vel@matrix),[actor])
    f,r=sensors.ego_rotate(*(player_vel@matrix),heading)
    return dict(rays=rays,visible_entities=entities,vitals=np.array([0.]*14+[f,r,0.],np.float32))
first = sense(np.eye(2))
assert first['visible_entities']['mask'].sum()==1
assert not np.array_equal(first['rays']['hit_type'],first['rays']['hit_type'][list(RAY_REFLECTION)])
mirror = reflect_observation(first)
actual = sense(transform)
for key in ('rays','visible_entities'):
    for field in mirror[key]:
        np.testing.assert_allclose(mirror[key][field],actual[key][field],rtol=2e-6,atol=2e-6)
np.testing.assert_allclose(mirror['vitals'],actual['vitals'],atol=2e-6)
# Pixel fields are carried through unchanged. The structured encoder never reads them.
payload={**first,'rgb':np.arange(12).reshape(2,2,3),'depth':np.arange(4).reshape(2,2)}
pixel=reflect_observation(payload)
np.testing.assert_array_equal(pixel['rgb'],payload['rgb'])
np.testing.assert_array_equal(pixel['depth'],payload['depth'])
x=torch.from_numpy(recorded[:4].copy()).requires_grad_()
reflect_packed(x).square().sum().backward()
assert torch.isfinite(x.grad).all()
print(f'PASS: official Sensors geometry, 216 action combinations, {observations} recorded structured frames, {windows} actual four-frame windows, padding, two reflections, and NumPy/Torch parity. No game initialized.')
