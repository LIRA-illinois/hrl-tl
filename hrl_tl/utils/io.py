def format_timesteps(timesteps: int) -> str:
    if timesteps >= 1_000_000:
        return f"{timesteps // 1_000_000}M"
    elif timesteps >= 1_000:
        return f"{timesteps // 1_000}K"
    else:
        return str(timesteps)
