import torch
from itertools import cycle
from ncodec.codec import TTSCodec
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from mira.utils import clear_cache


class MiraTTSStreaming:
    """MiraTTS with real chunked streaming via LMDeploy stream_infer"""

    def __init__(self, model_dir="YatharthS/MiraTTS", tp=1, enable_prefix_caching=True, cache_max_entry_count=0.2, dtype='bfloat16'):
        # Use TurboMind backend with same config as base MiraTTS class
        backend_config = TurbomindEngineConfig(
            cache_max_entry_count=cache_max_entry_count,
            tp=tp,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching
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

    def split_text_into_chunks(self, text, max_chunk_length=40):
        """
        Split text into smaller chunks for streaming (MeloTTS-style).
        Tries to split on sentence boundaries for natural speech.
        """
        import re

        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If adding this sentence would exceed max length, save current chunk
            if current_chunk and len(current_chunk) + len(sentence) > max_chunk_length:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence

        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # If no sentence boundaries found, split by length
        if not chunks:
            words = text.split()
            current_chunk = ""
            for word in words:
                if len(current_chunk) + len(word) + 1 > max_chunk_length:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = word
                else:
                    current_chunk = current_chunk + " " + word if current_chunk else word
            if current_chunk:
                chunks.append(current_chunk.strip())

        return chunks if chunks else [text]

    def stream_generate(self, text, context_tokens, chunk_size=40, reference_text=None):
        """Streaming generation using MeloTTS-style text chunking

        NOTE: MiraTTS doesn't have native streaming. We chunk the text
        and generate/stream each chunk sequentially (same as MeloTTS).

        Args:
            text: Text to synthesize
            context_tokens: Encoded reference audio
            chunk_size: Max characters per text chunk (default 150)
            reference_text: Transcript of reference audio
        """
        # Split text into chunks
        text_chunks = self.split_text_into_chunks(text, max_chunk_length=chunk_size)

        for i, text_chunk in enumerate(text_chunks):
            if not text_chunk.strip():
                continue

            # Generate full audio for this chunk
            formatted_prompt = self.codec.format_prompt(text_chunk, context_tokens, reference_text)
            response = self.pipe([formatted_prompt], gen_config=self.gen_config, do_preprocess=False)

            generated_text = response[0].text
            audio = self.codec.decode(generated_text, context_tokens)

            if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
                print(f"⚠️  Chunk {i+1}: No audio generated")
                continue

            # Yield the complete chunk audio
            audio_flat = audio.flatten()
            yield audio_flat

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
