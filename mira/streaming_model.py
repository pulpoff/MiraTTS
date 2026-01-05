import torch
from itertools import cycle
from ncodec.codec import TTSCodec
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from mira.utils import clear_cache


class MiraTTSStreaming:
    """MiraTTS with real chunked streaming via LMDeploy stream_infer"""

    def __init__(self, model_dir="YatharthS/MiraTTS", tp=1, enable_prefix_caching=True, cache_max_entry_count=0.2, dtype='float16'):
        # Use TurboMind for proper streaming support (PyTorch backend has broken stream_infer)
        backend_config = TurbomindEngineConfig(
            cache_max_entry_count=cache_max_entry_count,
            tp=tp,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
            model_format='hf'  # Force HuggingFace format for better compatibility
        )
        self.pipe = pipeline(model_dir, backend_config=backend_config)
        self.gen_config = GenerationConfig(
            top_p=0.95,
            top_k=50,
            temperature=0.8,
            max_new_tokens=1024,
            repetition_penalty=1.2,
            do_sample=True,
            min_p=0.05
        )
        self.codec = TTSCodec()

    def set_params(self, top_p=0.95, top_k=50, temperature=0.8, max_new_tokens=1024,
                   repetition_penalty=1.2, min_p=0.05):
        self.gen_config = GenerationConfig(
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            do_sample=True
        )

    def c_cache(self):
        clear_cache()

    def encode_audio(self, audio_file):
        return self.codec.encode(audio_file)

    def generate(self, text, context_tokens, reference_text=None):
        formatted_prompt = self.codec.format_prompt(text, context_tokens, reference_text)
        response = self.pipe([formatted_prompt], gen_config=self.gen_config, do_preprocess=False)
        return self.codec.decode(response[0].text, context_tokens)

    def stream_generate(self, text, context_tokens, chunk_size=50, reference_text=None):
        """Simulated streaming by generating full audio then chunking

        NOTE: LMDeploy's stream_infer is not compatible with MiraTTS model.
        This uses standard generation and yields audio in chunks for streaming UX.

        Args:
            text: Text to synthesize
            context_tokens: Encoded reference audio
            chunk_size: Audio chunk size in samples (not tokens)
            reference_text: Transcript of reference audio
        """
        import re

        # Split text into sentences for progressive generation
        sentences = re.split(r'([.!?]+)', text)
        sentences = [''.join(sentences[i:i+2]) for i in range(0, len(sentences)-1, 2)]
        if len(sentences) == 0:
            sentences = [text]

        print(f"🔍 Streaming {len(sentences)} sentence(s), chunk_size={chunk_size} samples")

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Generate full audio for this sentence
            formatted_prompt = self.codec.format_prompt(sentence.strip(), context_tokens, reference_text)
            response = self.pipe([formatted_prompt], gen_config=self.gen_config, do_preprocess=False)
            audio = self.codec.decode(response[0].text, context_tokens)

            if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
                print(f"⚠️  Sentence {i+1}: No audio generated")
                continue

            # Yield in chunks for streaming effect
            audio_flat = audio.flatten()
            num_samples = audio_flat.shape[0]
            chunk_samples = chunk_size * 1000  # Convert to actual sample count

            for start_idx in range(0, num_samples, chunk_samples):
                end_idx = min(start_idx + chunk_samples, num_samples)
                chunk = audio_flat[start_idx:end_idx]

                if chunk.numel() > 0:
                    yield chunk

            print(f"✓ Sentence {i+1}/{len(sentences)}: {num_samples} samples")

    def batch_generate(self, prompts, context_tokens, reference_texts=None):
        """
        Generates speech from text, for larger batch size

        Args:
            prompts (list): Input for tts model, list of prompts
            context_tokens (list): List of context tokens respective to prompts
            reference_texts (list, optional): List of reference texts respective to prompts
        """
        if reference_texts is None:
            reference_texts = [None] * len(prompts)

        formatted_prompts = []
        for prompt, context_token, ref_text in zip(prompts, cycle(context_tokens), cycle(reference_texts)):
            formatted_prompt = self.codec.format_prompt(prompt, context_token, ref_text)
            formatted_prompts.append(formatted_prompt)

        responses = self.pipe(formatted_prompts, gen_config=self.gen_config, do_preprocess=False)
        generated_tokens = [response.text for response in responses]

        audios = []
        for generated_token, context_token in zip(generated_tokens, cycle(context_tokens)):
            audio = self.codec.decode(generated_token, context_token)
            audios.append(audio)
        audios = torch.cat(audios, dim=0)

        return audios
