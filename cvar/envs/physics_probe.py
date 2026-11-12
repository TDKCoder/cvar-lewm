"""Probe which PushT physics knobs change the dynamics (Phase 0, block 6).

Runs the real stable-worldmodel PushT class. Each experiment compares the final T-block pose
(and the max position gap along the trajectory) against an unperturbed rollout with the same
seed and the same 60 random near-block actions. Output: the printed table, logged by
`python -m cvar.envs.physics_probe | tee runs/phase0-physics-probe/log.txt`.
"""
import numpy as np
import pymunk
from stable_worldmodel.envs.pusht.env import PushT
print("pymunk", pymunk.version)

def make(seed, mutate=None, **kw):
    env = PushT(**kw); env.reset(seed=seed)
    if mutate: mutate(env)
    return env

def rollout(seed, mutate=None, n=60, **kw):
    env = make(seed, mutate, **kw); rng = np.random.default_rng(1000 + seed); poses = []
    for t in range(n):
        a = rng.uniform(-1, 1, 2) * env.action_scale + np.array(env.agent.position)
        bp = np.array(env.block.position); a = np.clip(a, bp - 60, bp + 60)
        a = np.clip((a - np.array(env.agent.position)) / env.action_scale, -1, 1)
        env.step(a.astype(np.float32))
        poses.append([*env.block.position, env.block.angle])
    return np.array(poses), env

def diff(p, q):
    return f"pos {np.linalg.norm(p[-1,:2]-q[-1,:2]):7.2f} px  ang {abs(p[-1,2]-q[-1,2])*180/np.pi:6.2f} deg  max-traj-pos {np.linalg.norm(p[:,:2]-q[:,:2],axis=1).max():7.2f}"

env = make(0)
print("agent body type KINEMATIC:", env.agent.body_type == pymunk.Body.KINEMATIC, "| block mass", env.block.mass, "moment", round(env.block.moment, 1),
      "| shape frictions agent", [s.friction for s in env.agent.shapes], "block", [s.friction for s in env.block.shapes], "| damping", env.space.damping,
      "| block has .friction attr (stray python attr):", hasattr(env.block, "friction"))

# E1: velocity kick erased by damping=0? Move agent far away first so no contact.
env = make(0); env.agent.position = (30, 30); p0 = np.array(env.block.position)
env.block.velocity = (200.0, 0.0); env.step(np.zeros(2, np.float32)); p1 = np.array(env.block.position)
env2 = make(0); env2.agent.position = (30, 30); q0 = np.array(env2.block.position)
env2.block.apply_impulse_at_local_point((200.0, 0.0)); env2.step(np.zeros(2, np.float32))
print(f"E1 velocity kick 200 px/s -> block moved {np.linalg.norm(p1-p0):.3f} px; apply_impulse 200 -> {np.linalg.norm(np.array(env2.block.position)-q0):.3f} px")

# E2: temporary velocity_func keeps a set velocity through damping for one env step (10 substeps)
env = make(0); env.agent.position = (30, 30); p0 = np.array(env.block.position)
v = pymunk.Vec2d(200.0, 0.0)
def vf(body, gravity, damping, dt): body.velocity = v; body.angular_velocity = 0.5
env.block.velocity_func = vf; env.step(np.zeros(2, np.float32)); env.block.velocity_func = pymunk.Body.update_velocity
p1 = np.array(env.block.position); env.step(np.zeros(2, np.float32)); p2 = np.array(env.block.position)
print(f"E2 velocity_func 200 px/s for 10 substeps of 0.01 s -> moved {np.linalg.norm(p1-p0):.3f} px (expect 20); after restoring default it moved {np.linalg.norm(p2-p1):.3f} px more")

base, _ = rollout(0)
for s in (1, 2):
    b, _ = rollout(s); print(f"   (scale: seed 0 vs seed {s} random-action rollouts: {diff(base, b)})")
rep, _ = rollout(0); print("E0 determinism, same seed twice:", diff(base, rep))
for m in (0.25, 4.0):
    def mm(e, m=m): e.block.mass = e.block.mass * m
    def mi(e, m=m): e.block.mass = e.block.mass * m; e.block.moment = e.block.moment * m
    print(f"E3 mass x{m} only:          ", diff(base, rollout(0, mm)[0]))
    print(f"E3 mass+inertia x{m}:       ", diff(base, rollout(0, mi)[0]))
for mu in (0.3, 1.0):
    def fr(e, mu=mu):
        for s in e.block.shapes: s.friction = mu
        for s in e.agent.shapes: s.friction = 1.0
    print(f"E4 contact friction block {mu}, agent 1.0:", diff(base, rollout(0, fr)[0]))
for d in (1e-6, 1e-3, 0.1):
    def dm(e, d=d): e.space.damping = d
    print(f"E5 space.damping {d:g} (keeps {d**0.01:.3f}/substep):", diff(base, rollout(0, dm)[0]))
for cog in ((10, 0), (0, 15)):
    print(f"E6 block_cog {cog}:", diff(base, rollout(0, None, block_cog=cog)[0]))
