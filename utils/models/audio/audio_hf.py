"""Public AST audio-classification model fine-tuned on ESC-50 as a ReMaP student.

Wraps transformers AutoModelForAudioClassification + AutoFeatureExtractor so a public off-the-shelf
checkpoint plugs into the pipeline like our other students: forward(waveform) -> softmax probs (deepgini/
maxp; argmax == logits argmax). The 256/768-d penultimate for ReMaP/DuFP kNN is the output of `penultimate`
(a registered nn.Module that mean-pools hidden_states[-1] over the time axis), consumed via a standard
forward-hook, with no forward_feature() shortcut needed. The loader
yields 48 kHz mono waveforms (CLAP-native); here we resample to the model's own sr and run its feature
extractor. We do NOT train -- third-party weights, demonstrating ReMaP audits public students. librosa/
transformers imported lazily so importing this module never requires them on non-audio runs.
"""
import torch
import torch.nn as nn

AST_REVISION = 'fb2443ee3c362d6683a8f493eab2d15598360fb6'


class _MeanTimePool(nn.Module):
    """Mean-pool over the time axis of a transformer's last hidden state ([B,T,H] -> [B,H]).

    A registered submodule so its forward-hook output IS the penultimate feature the ReMaP/DuFP kNN
    consumes -- the audio analog of BERT's hf_model.dropout hook target.
    """
    def forward(self, x):
        return x.mean(dim=1)


class AudioHFClassifier(nn.Module):
    def __init__(self, repo_or_dir, device, src_sr=48000):
        super().__init__()
        import librosa
        from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
        self._librosa = librosa
        self.fe = AutoFeatureExtractor.from_pretrained(repo_or_dir, revision=AST_REVISION)
        self.model = AutoModelForAudioClassification.from_pretrained(
            repo_or_dir, revision=AST_REVISION).eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.sr = self.fe.sampling_rate  # the model's expected sample rate (e.g. 16 kHz)
        self.src_sr = src_sr  # the loader's sample rate (48 kHz, CLAP-native)
        self.penultimate = _MeanTimePool()  # hookable [B,H] penultimate for ReMaP/DuFP kNN

    def _featurize(self, wav):
        """wav [B, T] @ src_sr -> the model's input tensors on wav.device (resample + feature-extract)."""
        arrs = [self._librosa.resample(w.detach().cpu().numpy().astype('float32'),
                                       orig_sr=self.src_sr, target_sr=self.sr) for w in wav]
        inputs = self.fe(arrs, sampling_rate=self.sr, return_tensors='pt', padding=True)
        return {k: v.to(wav.device) for k, v in inputs.items()}

    # forward_feature shortcut removed; hook `penultimate` (nn.Module returning hidden_states[-1].mean(dim=1))
    # def forward_feature(self, wav):
    #     out = self.model(**self._featurize(wav), output_hidden_states=True)
    #     return out.hidden_states[-1].mean(dim=1)

    @torch.no_grad()
    def forward(self, wav):
        out = self.model(**self._featurize(wav), output_hidden_states=True)
        _ = self.penultimate(out.hidden_states[-1])  # fires the forward-hook that ReMaP/DuFP register
        return torch.softmax(out.logits, dim=1)  # probs for deepgini; argmax == logits argmax


def load_audio_hf_classifier(repo_or_dir, device):
    return AudioHFClassifier(repo_or_dir, device)
