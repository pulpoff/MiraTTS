import torch
from itertools import cycle
from ncodec.codec import TTSCodec
from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from mira.utils import clear_cache


class MiraTTSStreaming:
    """MiraTTS with real chunked streaming via LMDeploy stream_infer"""

    def __init__(self, model_dir="YatharthS/MiraTTS", tp=1, enable_prefix_caching=True, cache_max_entry_count=0.2, dtype='float16'):
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

    def stream_generate(self, text, context_tokens, chunk_size=50, reference_text=None):
        """Stream audio chunks as tokens are generated

        Args:
            text: Text to synthesize
            context_tokens: Encoded reference audio
            chunk_size: Tokens to accumulate before decoding (lower=faster, higher=efficient)
            reference_text: Transcript of reference audio
        """
        formatted_prompt = self.codec.format_prompt(text, context_tokens, reference_text)
        print(f"🔍 DEBUG: Input text='{text[:50]}...' ref_text='{reference_text[:30] if reference_text else None}'")

        accumulated_tokens = ""
        previous_audio_length = 0
        tokens_since_decode = 0
        iteration_count = 0
        total_tokens_generated = 0

        for response in self.pipe.stream_infer([formatted_prompt], gen_config=self.gen_config, do_preprocess=False):
            iteration_count += 1

            # LMDeploy stream_infer returns FULL accumulated text, but may be unstable
            # Track growth properly by checking actual content
            current_text = response.text

            # Only update if we have more tokens than before
            if len(current_text) > len(accumulated_tokens):
                prev_length = len(accumulated_tokens)
                accumulated_tokens = current_text
                new_tokens = len(accumulated_tokens) - prev_length
                tokens_since_decode += new_tokens
                total_tokens_generated += new_tokens

                if iteration_count <= 3 or iteration_count % 10 == 0:
                    print(f"🔍 Iter {iteration_count}: new_tokens={new_tokens}, total={total_tokens_generated}, accumulated_len={len(accumulated_tokens)}")
            elif iteration_count <= 3 or iteration_count % 10 == 0:
                print(f"⚠️  Iter {iteration_count}: text shrunk or stayed same (current={len(current_text)}, accumulated={len(accumulated_tokens)})")

            should_decode = tokens_since_decode >= chunk_size or response.finish_reason is not None

            if should_decode and accumulated_tokens:
                try:
                    full_audio = self.codec.decode(accumulated_tokens, context_tokens)

                    if isinstance(full_audio, torch.Tensor) and full_audio.numel() > 0:
                        current_length = full_audio.shape[0]

                        if current_length > previous_audio_length:
                            new_audio = full_audio[previous_audio_length:]
                            previous_audio_length = current_length
                            tokens_since_decode = 0

                            if new_audio.numel() > 0:
                                print(f"✓ Yielding audio chunk: {new_audio.shape[0]} samples")
                                yield new_audio
                        else:
                            print(f"⚠️  Audio not growing: current={current_length}, prev={previous_audio_length}")
                    else:
                        print(f"⚠️  Decoded audio is empty or invalid type")
                except Exception as e:
                    print(f"⚠️  Decode error at iteration {iteration_count}: {e}")
                    continue

            if response.finish_reason is not None:
                print(f"🏁 Stream finished: iterations={iteration_count}, total_tokens={total_tokens_generated}, finish_reason={response.finish_reason}")
                if iteration_count == 0 or total_tokens_generated == 0:
                    print(f"⚠️  Streaming ended with no tokens generated!")
                break

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
