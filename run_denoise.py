from src.denoisers import Denoiser
import warnings
import hydra
import torchaudio
from hydra.utils import instantiate
from pathlib import Path
from tqdm import tqdm 

warnings.filterwarnings("ignore", category=UserWarning)

@hydra.main(version_base=None, config_path="configs", config_name="denoise")
def main(config):
    """
    Main script for denoising. Instantiates the denoiser class.
    Runs the chosen denoiser on the input folder and outputs it in the outpur folder.
    Input folder should consist of *.mp3, *.wav, *.flac files

    Args:
        config (DictConfig): hydra config.
    """

    assert config.denoise_method in ['simple', 'window'], 'denoising method must be either "simple" or "window"'
    allowed_extensions = {".mp3", ".flac", ".wav"}

    input_dir = Path(config.dirs.input_dir).resolve()
    output_dir = Path(config.dirs.output_dir).resolve() / config.denoise_method
    output_dir.mkdir(parents=True, exist_ok=True)
    
    denoiser = instantiate(config.denoiser)
    if config.denoise_method == 'simple':
      denoise_fn = denoiser.simple_denoiser
    else:
      denoise_fn = denoiser.window_denoiser
    
    for path in tqdm(input_dir.rglob("*"), desc='Denoising files...'):
        if path.suffix.lower() in allowed_extensions:
            denoised_audio, sr = denoise_fn(path)

            rel_path = path.relative_to(input_dir)
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            torchaudio.save(out_path, denoised_audio, sr)


    print('Denoising complete')

            
if __name__ == "__main__":
    main()
