import argparse
import os
import sys
import torch
from musc.model import PretrainedModel
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    parser = argparse.ArgumentParser(description="Convert WAV to MIDI using MUSC model.")
    parser.add_argument("input_wav", help="Path to the input WAV file")
    parser.add_argument("output_mid", help="Path to the output MIDI file")
    
    args = parser.parse_args()
    
    input_wav = os.path.expanduser(args.input_wav)
    output_mid = os.path.expanduser(args.output_mid)
    
    if not os.path.exists(input_wav):
        print(f"Error: Input file '{input_wav}' does not exist.")
        sys.exit(1)
        
    print(f"Loading model...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # Map location to CPU if CUDA is not available is handled internally by torch.load 
        # but the model class .to(device) handles moving it.
        # However, looking at the code of PretrainedModel in musc/model.py:
        # self.load_state_dict(torch.load(os.path.join(package_dir, filename)))
        # It doesn't specify map_location. So it might fail if the model was saved with CUDA and we are on CPU.
        # We might need to monkeypatch torch.load or modify the model code if it fails.
        # Let's try to instantiate it.
        
        # Monkey patch torch.load if device is cpu to avoid issues with loading cuda weights on cpu
        if device == 'cpu':
            original_load = torch.load
            def map_location_load(*args, **kwargs):
                if 'map_location' not in kwargs:
                    kwargs['map_location'] = torch.device('cpu')
                return original_load(*args, **kwargs)
            torch.load = map_location_load
            
        model = PretrainedModel(instrument='violin').to(device)
        
        # Restore torch.load just in case
        if device == 'cpu':
             torch.load = original_load
             
    except Exception as e:
        print(f"Error loading model: {e}")
        # Retry with explicit cpu mapping if it failed and we are on cpu, 
        # although the monkeypatch above should handle it.
        sys.exit(1)
        
    print(f"Transcribing '{input_wav}' to '{output_mid}'...")
    
    try:
        # The transcribe method returns a pretty_midi object
        midi = model.transcribe(input_wav, batch_size=32, postprocessing='spotify')
        midi.write(output_mid)
        print(f"Successfully saved MIDI to '{output_mid}'")
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import os
    main()

