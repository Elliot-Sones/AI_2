#!/usr/bin/env python3
"""Play against a classic CPU policy with its training observation wrappers.

Controls: WASD to move, SPACE to jump, H/J/K/L/G for attacks.
"""

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from tools.visualization.watch_games import FrameStack, load_training, make_game_env


class UserInputAgent:
    def __init__(self, pygame):
        self.pygame = pygame
        self.initialized = False

    def get_env_info(self, env):
        self.act_helper = env.act_helper
        self.initialized = True

    def predict(self, obs):
        action = self.act_helper.zeros()
        keys = self.pygame.key.get_pressed()
        for key in ("w", "a", "s", "d", "space", "h", "j", "k", "l", "g"):
            code = getattr(self.pygame, "K_SPACE" if key == "space" else f"K_{key}")
            if keys[code]:
                action = self.act_helper.press_keys([key], action)
        return action


def play_match(model, *, frame_stack=4, seconds=180):
    import pygame
    pygame.init()
    env = None
    try:
        player = UserInputAgent(pygame)
        env = make_game_env(player, max_steps=30 * seconds)
        obs, _ = env.reset()
        stack = FrameStack(frame_stack, obs.shape)
        stacked_obs = stack.reset(obs)
        screen = pygame.display.set_mode((720, 480))
        pygame.display.set_caption("AI Squared - Player vs AI")
        clock = pygame.time.Clock()
        for _ in range(30 * seconds):
            events = pygame.event.get()
            if any(event.type == pygame.QUIT or
                   (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE)
                   for event in events):
                break
            action, _ = model.predict(stacked_obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            stacked_obs = stack.step(obs)
            screen.blit(pygame.surfarray.make_surface(env.render()), (0, 0))
            pygame.display.flip()
            clock.tick(30)
            if terminated or truncated:
                break
        raw_env = env.unwrapped.raw_env
        # The model is training player 0; the human is its opponent (player 1).
        return raw_env.get_stats(1), raw_env.get_stats(0)
    finally:
        try:
            if env is not None:
                env.close()
        finally:
            pygame.quit()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Play against a classic CPU policy")
    parser.add_argument("--model", "-m", required=True, help="Path to PPO checkpoint")
    parser.add_argument("--frame-stack", type=int, default=4, help="Match the training configuration")
    parser.add_argument("--seconds", type=int, default=180, help="Maximum match duration")
    args = parser.parse_args(argv)
    if args.frame_stack < 1 or args.seconds < 1:
        parser.error("--frame-stack and --seconds must be positive")
    if not Path(args.model).is_file():
        parser.error(f"model not found: {args.model}")
    load_training()
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device="cpu")
    print("You are player 2. WASD: move; SPACE: jump; H/J/K/L/G: attacks. ESC: quit.")
    try:
        human, agent = play_match(model, frame_stack=args.frame_stack, seconds=args.seconds)
    except KeyboardInterrupt:
        print("Match cancelled.")
        return
    print(f"Lives: you {human.lives_left}, AI {agent.lives_left}")
    print(f"Damage dealt: you {human.damage_done:.1f}, AI {agent.damage_done:.1f}")


if __name__ == "__main__":
    main()
