import torch
import re
from itertools import cycle
from ncodec.codec import TTSCodec
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from mira.utils import clear_cache


class MiraTTSStreaming:
    """MiraTTS with token-level streaming via PyTorch backend stream_infer"""

    def __init__(self, model_dir="YatharthS/MiraTTS", tp=1, enable_prefix_caching=True, cache_max_entry_count=0.2, dtype='bfloat16'):
        # Use PyTorch backend for stream_infer support
        backend_config = PytorchEngineConfig(
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

    def split_text_into_chunks(self, text, max_chunk_length=25):
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

    def stream_generate(self, text, context_tokens, chunk_size=50, reference_text=None):
        """Token-level streaming generation using PyTorch backend stream_infer

        Attempts true token-level streaming by accumulating tokens and decoding
        them in chunks. Falls back to text chunking if streaming fails.

        Args:
            text: Text to synthesize
            context_tokens: Encoded reference audio
            chunk_size: Number of tokens to accumulate before decoding (default 50)
            reference_text: Transcript of reference audio
        """
        formatted_prompt = self.codec.format_prompt(text, context_tokens, reference_text)

        try:
            # Attempt token-level streaming with PyTorch backend
            accumulated_tokens = []
            token_count = 0
            chunk_yield_count = 0

            for output in self.pipe.stream_infer([formatted_prompt], gen_config=self.gen_config, do_preprocess=False):
                token_text = output.text

                # Extract new token(s) from output
                if token_text:
                    accumulated_tokens.append(token_text)
                    token_count += 1

                    # Decode every chunk_size tokens
                    if token_count >= chunk_size:
                        full_token_sequence = ''.join(accumulated_tokens)

                        # Check if we have valid speech tokens
                        if '<|speech_token_' in full_token_sequence:
                            try:
                                audio = self.codec.decode(full_token_sequence, context_tokens)

                                if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                                    chunk_yield_count += 1
                                    yield audio.flatten()
                                    accumulated_tokens = []
                                    token_count = 0
                            except Exception as e:
                                # Continue accumulating if decode fails
                                pass

            # Decode remaining tokens
            if accumulated_tokens:
                full_token_sequence = ''.join(accumulated_tokens)
                if '<|speech_token_' in full_token_sequence:
                    try:
                        audio = self.codec.decode(full_token_sequence, context_tokens)
                        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                            chunk_yield_count += 1
                            yield audio.flatten()
                    except Exception as e:
                        print(f"WARNING: Failed to decode final tokens: {e}")

        except Exception as e:
            print(f"WARNING: Token-level streaming failed: {e}")
            print(f"INFO: Falling back to text chunking approach")

            # Fallback: text chunking approach (MeloTTS-style)
            text_chunks = self.split_text_into_chunks(text, max_chunk_length=25)

            for i, text_chunk in enumerate(text_chunks):
                if not text_chunk.strip():
                    continue

                formatted_prompt = self.codec.format_prompt(text_chunk, context_tokens, reference_text)
                response = self.pipe([formatted_prompt], gen_config=self.gen_config, do_preprocess=False)

                generated_text = response[0].text
                audio = self.codec.decode(generated_text, context_tokens)

                if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
                    print(f"WARNING: Chunk {i+1}: No audio generated")
                    continue

                yield audio.flatten()

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
