## My own Lil' Denoiser 

This repository contains two variations of spectral denoisers: a simple one which estimates noise by taking the global minimum over the entire audio signal, and a window-based one that performs noise estimation locally using sliding windows in the spectral domain.

### Key features 

- **Two denoiser strategies** (simple, windowed)
- **Hydra configuration support** (for denoising parameters and input/output folders)
- **Noise generating functions** (white, pink, band)

### How to use

Install all dependencies:
```bash
pip install -r ./requirements.txt
```

Make sure that before denoising you have all audios are *.mp3, *.flac or *.wav.

To denoise your audios, run:
```bash
python run_denoise.py -cn=denoise denoise_method=YOUR_DENOISE_METHOD dirs.input_dir=YOUR_INPUT_DIRECTORY dirs.output_dir=YOUR_OUTPUT_DIRECTORY
```

You can change denoise method, its strenth (alpha), and input/output paths via hydra in terminal.

## Credits

Made by htutb

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
