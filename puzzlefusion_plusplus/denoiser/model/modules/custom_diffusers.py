from diffusers import DDPMScheduler
import torch
import math

def betas_for_alpha_bar(
    num_diffusion_timesteps=1000,
    max_beta=0.999,
    alpha_transform_type="piece_wise",
):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
        alpha_transform_type (`str`, *optional*, default to `cosine`): the type of noise schedule for alpha_bar.
                     Choose from `cosine` or `exp`

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """
    if alpha_transform_type == "cosine":

        def alpha_bar_fn(t):
            return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    elif alpha_transform_type == "exp":

        def alpha_bar_fn(t):
            return math.exp(t * -12.0)
        
    elif alpha_transform_type == "piece_wise":
        def alpha_bar_fn(t):
            t = t * 1000
            if t <= 700:
                # Quadratic decrease from 1 to 0.9 between x = 0 to 700
                return 1 - 0.1 * (t / 700)**2
            else:
                # Quadratic decrease from 0.9 to 0 between x = 700 to 1000
                return 0.9 * (1 - ((t - 700) / 300)**2)

    else:
        raise ValueError(f"Unsupported alpha_tranform_type: {alpha_transform_type}")

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar_fn(t2) / alpha_bar_fn(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)



class PiecewiseScheduler(DDPMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.betas = betas_for_alpha_bar(
            alpha_transform_type = "piece_wise"
        )
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)